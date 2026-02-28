"""Unit tests for the Code Expert tools.

Run:  uv run python -m tests.test_code_tools
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from experts.code.tools.code import code_tool
from experts.code.tools.text import text_tool

TOTAL = 0
PASSED = 0


def check(label, result, key, expected):
    global TOTAL, PASSED
    TOTAL += 1
    actual = result.get(key)
    ok = actual == expected
    PASSED += ok
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    if not ok:
        print(f"         Expected {key}={expected!r}")
        print(f"         Got      {key}={actual!r}")
    return ok


def check_in(label, result, key, substring):
    """Check that result[key] contains substring."""
    global TOTAL, PASSED
    TOTAL += 1
    actual = result.get(key, "")
    ok = substring in str(actual)
    PASSED += ok
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    if not ok:
        print(f"         Expected '{substring}' in {key}")
        print(f"         Got      {key}={actual!r}")
    return ok


# ── code_tool: check ──────────────────────────────────

def test_code_check():
    print("\n--- code_tool (check) ------------------------------------")
    check("valid syntax",     code_tool("x = 1 + 2", "check"),              "valid", True)
    check("invalid syntax",   code_tool("def foo(", "check"),               "valid", False)
    check("blocked import",   code_tool("import os", "check"),              "valid", False)
    check("blocked from-import", code_tool("from subprocess import run", "check"), "valid", False)


# ── code_tool: ast_inspect ────────────────────────────

def test_code_inspect():
    print("\n--- code_tool (ast_inspect) -------------------------------")
    code = """
def greet(name):
    return f"Hello, {name}!"

class Calculator:
    def add(self, a, b):
        return a + b
"""
    r = code_tool(code, "ast_inspect")
    check("has functions",    r, "verified", True)
    check("function count",   {"count": len(r.get("functions", []))}, "count", 2)
    check("class count",      {"count": len(r.get("classes", []))},   "count", 1)


# ── code_tool: run ────────────────────────────────────

def test_code_run():
    print("\n--- code_tool (run) --------------------------------------")
    check("print output",     code_tool('print("hello")', "run"),           "stdout", "hello\n")
    check("exit code 0",      code_tool('print(42)', "run"),                "exit_code", 0)
    check("runtime error",    code_tool('1/0', "run"),                      "exit_code", 1)
    check_in("stderr on err", code_tool('1/0', "run"),                      "stderr", "ZeroDivisionError")
    check("blocked os",       code_tool('import os; os.listdir(".")', "run"), "exit_code", 1)
    check("timeout",          code_tool('import time; time.sleep(30)', "run", timeout=2), "exit_code", -1)
    check("stdin",            code_tool('x = input(); print(f"Got: {x}")', "run", stdin_data="test"), "stdout", "Got: test\n")


# ── text_tool: regex_find ─────────────────────────────

def test_text_regex():
    print("\n--- text_tool (regex) ------------------------------------")
    check("find digits",      text_tool("abc 123 def 456", "regex_find", pattern=r"\d+"), "count", 2)
    r = text_tool("abc 123 def 456", "regex_find", pattern=r"\d+")
    check("matches",          r, "matches", ["123", "456"])

    r = text_tool("Hello World", "regex_replace", pattern=r"World", replacement="Python")
    check("replace",          r, "result", "Hello Python")
    check("replace count",    r, "replacements", 1)

    check("invalid regex",    text_tool("abc", "regex_find", pattern=r"["), "error", "Invalid regex: unterminated character set at position 0")


# ── text_tool: count ──────────────────────────────────

def test_text_count():
    print("\n--- text_tool (count) ------------------------------------")
    r = text_tool("hello world", "count")
    check("char count",       r, "characters", 11)
    check("word count",       r, "words", 2)
    check("line count",       r, "lines", 1)

    check("regex count",      text_tool("a1b2c3", "count", pattern=r"\d"), "count", 3)


# ── text_tool: split ──────────────────────────────────

def test_text_split():
    print("\n--- text_tool (split) ------------------------------------")
    r = text_tool("a,b,c", "split", pattern=",")
    check("split by comma",   r, "parts", ["a", "b", "c"])
    check("split count",      r, "count", 3)


# ── text_tool: json_parse ─────────────────────────────

def test_text_json():
    print("\n--- text_tool (json_parse) --------------------------------")
    r = text_tool('{"name": "Roy", "age": 22}', "json_parse")
    check("json type",        r, "type", "object")
    check("json keys",        r, "keys", ["name", "age"])

    r = text_tool('[1, 2, 3]', "json_parse")
    check("array type",       r, "type", "array")
    check("array len",        r, "length", 3)

    check_in("invalid json",  text_tool("{bad}", "json_parse"), "error", "Invalid JSON")


# ── text_tool: diff ───────────────────────────────────

def test_text_diff():
    print("\n--- text_tool (diff) -------------------------------------")
    r = text_tool("line1\nline2\nline3", "diff", text_b="line1\nmodified\nline3")
    check("has diff",         r, "verified", True)
    check("added lines",     r, "added", 1)
    check("removed lines",   r, "removed", 1)


# ── Main ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  ReasonForge — Code Expert Tests")
    print("=" * 52)

    test_code_check()
    test_code_inspect()
    test_code_run()
    test_text_regex()
    test_text_count()
    test_text_split()
    test_text_json()
    test_text_diff()

    print("\n" + "=" * 52)
    print(f"  {PASSED}/{TOTAL} passed")
    if PASSED == TOTAL:
        print("  All tests passed!")
    else:
        print(f"  {TOTAL - PASSED} test(s) FAILED.")
    print("=" * 52)
