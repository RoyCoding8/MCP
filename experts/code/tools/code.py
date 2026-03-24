"""Sandboxed Python code execution with AST inspection."""

import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

OPERATIONS = {"run", "check", "ast_inspect"}

_BLOCKED_IMPORTS = {
    "os",
    "shutil",
    "subprocess",
    "multiprocessing",
    "threading",
    "ctypes",
    "signal",
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "webbrowser",
    "pathlib",
    "glob",
    "tempfile",
    "importlib",
    "pickle",
    "shelve",
    "marshal",
    "code",
    "codeop",
    "compile",
    "compileall",
}

_BLOCKED_ATTRS = {
    "system",
    "popen",
    "exec",
    "eval",
    "execfile",
    "rmtree",
    "remove",
    "unlink",
    "rename",
    "__import__",
    "__subclasses__",
    "__globals__",
    "__builtins__",
    "__bases__",
    "__mro__",
}

_BLOCKED_CALLS = {
    "open",
    "input",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}

_MAX_OUTPUT = 2000
_MAX_CODE_CHARS = 12000
_MAX_STDIN_CHARS = 4000
_MIN_TIMEOUT = 1
_MAX_TIMEOUT = 15
_ISOLATION_MODES = {"auto", "docker", "process"}
_DEFAULT_DOCKER_IMAGE = "python:3.11-alpine"


def _scan_imports(tree: ast.AST) -> list[str]:
    """Return list of blocked import/attribute violations in an AST."""
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    violations.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    violations.append(f"blocked import: from {node.module}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                violations.append(f"blocked attribute: .{node.attr}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
                violations.append(f"blocked builtin: {node.func.id}()")
    return violations


def _sanitize_timeout(timeout: int) -> int:
    try:
        timeout_i = int(timeout)
    except (TypeError, ValueError):
        timeout_i = 10
    return max(_MIN_TIMEOUT, min(_MAX_TIMEOUT, timeout_i))


def _isolation_mode() -> str:
    mode = os.environ.get("RF_CODE_TOOL_ISOLATION", "auto").strip().lower()
    return mode if mode in _ISOLATION_MODES else "auto"


def _docker_image() -> str:
    return os.environ.get("RF_CODE_TOOL_DOCKER_IMAGE", _DEFAULT_DOCKER_IMAGE).strip()


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def _subprocess_kwargs(timeout: int) -> dict:
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ.get("PATH", ""),
    }
    kwargs: dict = {"env": env}

    if os.name != "nt":

        def _apply_limits():
            try:
                import resource

                mem_bytes = 256 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            except Exception:
                return

        kwargs["preexec_fn"] = _apply_limits

    return kwargs


def _run_python_process(script_path: Path, timeout: int, stdin_data: str):
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-u", str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin_data or None,
        cwd=str(script_path.parent),
        **_subprocess_kwargs(timeout),
    )


def _run_python_docker(script_path: Path, timeout: int, stdin_data: str):
    host_dir = str(script_path.parent.resolve())
    mount = f"{host_dir}:/workspace:ro"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "0.5",
        "--memory",
        "256m",
        "--pids-limit",
        "64",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        "-v",
        mount,
        "-w",
        "/tmp",
        _docker_image(),
        "python",
        "-I",
        "-S",
        "-B",
        "-u",
        "/workspace/snippet.py",
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 3,
        input=stdin_data or None,
    )


def _execute_python(script_path: Path, timeout: int, stdin_data: str):
    mode = _isolation_mode()

    docker_ok = False
    if mode in {"auto", "docker"}:
        docker_ok = _docker_available()

    if mode in {"auto", "docker"} and docker_ok:
        return _run_python_docker(script_path, timeout, stdin_data), "docker", None

    warning = None
    if mode == "docker" and not docker_ok:
        warning = "Docker unavailable; fell back to process isolation."

    return _run_python_process(script_path, timeout, stdin_data), "process", warning


def code_tool(
    code: str, operation: str = "run", timeout: int = 10, stdin_data: str = ""
) -> dict:
    """Sandboxed Python code execution tool.

    Use for running Python code, syntax checking, and code structure inspection.
    Operations: run (execute code), check (syntax-only), ast_inspect (structure analysis).
    Code is executed in an isolated subprocess with timeout and blocked dangerous imports.
    """
    try:
        if operation not in OPERATIONS:
            return {
                "error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}",
                "verified": True,
            }

        timeout = _sanitize_timeout(timeout)

        if stdin_data is None:
            stdin_data = ""
        elif not isinstance(stdin_data, str):
            stdin_data = str(stdin_data)

        if len(stdin_data) > _MAX_STDIN_CHARS:
            return {
                "error": f"stdin_data too large ({len(stdin_data)} chars). Max {_MAX_STDIN_CHARS}.",
                "verified": True,
            }

        code = textwrap.dedent(code).strip()
        if not code:
            return {"error": "Code cannot be empty.", "verified": True}
        if len(code) > _MAX_CODE_CHARS:
            return {
                "error": f"Code too large ({len(code)} chars). Max {_MAX_CODE_CHARS}.",
                "verified": True,
            }

        if operation == "check":
            try:
                tree = ast.parse(code)
                violations = _scan_imports(tree)
                if violations:
                    return {"valid": False, "blocked": violations, "verified": True}
                return {"valid": True, "verified": True}
            except SyntaxError as e:
                return {
                    "valid": False,
                    "error": f"SyntaxError: {e.msg}",
                    "line": e.lineno,
                    "offset": e.offset,
                    "verified": True,
                }

        elif operation == "ast_inspect":
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {
                    "error": f"SyntaxError: {e.msg}",
                    "line": e.lineno,
                    "verified": True,
                }

            functions, classes, imports = [], [], []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    functions.append(
                        {"name": node.name, "args": args, "line": node.lineno}
                    )
                elif isinstance(node, ast.AsyncFunctionDef):
                    args = [a.arg for a in node.args.args]
                    functions.append(
                        {
                            "name": f"async {node.name}",
                            "args": args,
                            "line": node.lineno,
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    classes.append(
                        {"name": node.name, "methods": methods, "line": node.lineno}
                    )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "total_lines": len(code.splitlines()),
                "verified": True,
            }

        elif operation == "run":
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {
                    "error": f"SyntaxError: {e.msg}",
                    "line": e.lineno,
                    "exit_code": 1,
                    "verified": True,
                }

            violations = _scan_imports(tree)
            if violations:
                return {
                    "error": "Blocked for safety",
                    "blocked": violations,
                    "exit_code": 1,
                    "verified": True,
                }

            with tempfile.TemporaryDirectory(prefix="reasonforge_") as tmp_dir:
                tmp_path = Path(tmp_dir) / "snippet.py"
                tmp_path.write_text(code, encoding="utf-8")

                try:
                    result, isolation_mode, warning = _execute_python(
                        tmp_path, timeout, stdin_data
                    )
                    stdout = result.stdout[:_MAX_OUTPUT] if result.stdout else ""
                    stderr = result.stderr[:_MAX_OUTPUT] if result.stderr else ""
                    out = {
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": result.returncode,
                        "isolation_mode": isolation_mode,
                        "timeout_s": timeout,
                        "verified": True,
                    }
                    if warning:
                        out["warning"] = warning
                    if (
                        len(result.stdout or "") > _MAX_OUTPUT
                        or len(result.stderr or "") > _MAX_OUTPUT
                    ):
                        out["truncated"] = True
                    return out
                except subprocess.TimeoutExpired:
                    return {
                        "error": f"Execution timed out ({timeout}s)",
                        "exit_code": -1,
                        "isolation_mode": _isolation_mode(),
                        "timeout_s": timeout,
                        "verified": True,
                    }

        return {"error": "Unhandled operation path.", "verified": True}

    except Exception as e:
        return {"error": str(e), "operation": operation}

    return {"error": "Unexpected execution fallthrough.", "verified": True}
