"""Code tool — sandboxed Python execution with AST inspection.

Operations: run, check, ast_inspect.
Uses subprocess with timeout for safe execution.
Hard AST-based import blocking prevents dangerous operations.
"""

import ast
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

OPERATIONS = {"run", "check", "ast_inspect"}

# Imports/attributes that are blocked before execution
_BLOCKED_IMPORTS = {
    "os", "shutil", "subprocess", "multiprocessing", "threading",
    "ctypes", "signal", "socket", "http", "urllib", "requests",
    "ftplib", "smtplib", "telnetlib", "webbrowser",
    "pathlib", "glob", "tempfile", "importlib",
    "pickle", "shelve", "marshal",
    "code", "codeop", "compile", "compileall",
}

_BLOCKED_ATTRS = {
    "system", "popen", "exec", "eval", "execfile",
    "rmtree", "remove", "unlink", "rename",
    "__import__", "__subclasses__", "__globals__",
}

_MAX_OUTPUT = 2000  # Max chars of stdout/stderr to return


def _scan_imports(tree: ast.AST) -> list[str]:
    """Scan an AST for blocked imports. Returns list of violations."""
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
            if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile", "__import__"):
                violations.append(f"blocked builtin: {node.func.id}()")
    return violations


def code_tool(code: str, operation: str = "run",
              timeout: int = 10, stdin_data: str = "") -> dict:
    """Sandboxed Python code execution tool.

    Use for running Python code, syntax checking, and code structure inspection.
    Operations: run (execute code), check (syntax-only), ast_inspect (structure analysis).
    Code is executed in an isolated subprocess with timeout and blocked dangerous imports.
    """
    try:
        code = textwrap.dedent(code).strip()

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
                return {"error": f"SyntaxError: {e.msg}", "line": e.lineno, "verified": True}

            functions = []
            classes = []
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [a.arg for a in node.args.args]
                    functions.append({"name": node.name, "args": args, "line": node.lineno})
                elif isinstance(node, ast.AsyncFunctionDef):
                    args = [a.arg for a in node.args.args]
                    functions.append({"name": f"async {node.name}", "args": args, "line": node.lineno})
                elif isinstance(node, ast.ClassDef):
                    methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    classes.append({"name": node.name, "methods": methods, "line": node.lineno})
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
            # Parse and scan for blocked imports first
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {"error": f"SyntaxError: {e.msg}", "line": e.lineno, "exit_code": 1, "verified": True}

            violations = _scan_imports(tree)
            if violations:
                return {"error": "Blocked for safety", "blocked": violations, "exit_code": 1, "verified": True}

            # Write to temp file and execute in subprocess
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                tmp_path = f.name

            try:
                result = subprocess.run(
                    [sys.executable, "-u", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    input=stdin_data or None,
                    cwd=tempfile.gettempdir(),
                )
                stdout = result.stdout[:_MAX_OUTPUT] if result.stdout else ""
                stderr = result.stderr[:_MAX_OUTPUT] if result.stderr else ""

                out = {"stdout": stdout, "stderr": stderr, "exit_code": result.returncode, "verified": True}
                if len(result.stdout or "") > _MAX_OUTPUT:
                    out["truncated"] = True
                return out

            except subprocess.TimeoutExpired:
                return {"error": f"Execution timed out ({timeout}s)", "exit_code": -1, "verified": True}
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        else:
            return {"error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}"}

    except Exception as e:
        return {"error": str(e), "operation": operation}
