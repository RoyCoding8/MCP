import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from experts.math.tools.algebra import math_tool
from experts.math.tools.calculus import calculus_tool
from experts.math.tools.matrix import matrix_tool
from experts.math.tools.statistics import statistics_tool
from experts.code.tools.code import code_tool

TOTAL = 0
PASSED = 0


def check(label, result, key, expected):
    global TOTAL, PASSED
    TOTAL += 1
    actual = result.get(key)
    def norm(v):
        if isinstance(v, list):
            try: return sorted(v, key=lambda x: float(x) if x is not None else 0)
            except: return sorted(str(x) for x in v)
        return v
    ok = norm(actual) == norm(expected)
    PASSED += ok
    print(f"{'[PASS]' if ok else '[FAIL]'} {label}")
    if not ok:
        print(f"Expected {key}={expected!r}")
        print(f"Got      {key}={actual!r}")


def test_math():
    print("\n--- math_tool ---")
    check("compute", math_tool("347*892"), "result", "309524")
    check("solve", math_tool("x**2-5*x+6", "solve"), "solutions", ["2", "3"])
    check("simplify", math_tool("(x**2-1)/(x-1)", "simplify"), "result", "x + 1")
    check("factor", math_tool("x**2-5*x+6", "factor"), "result", "(x - 3)*(x - 2)")
    check("gcd", math_tool("12, 18", "gcd"), "result", "6")


def test_calculus():
    print("\n--- calculus_tool ---")
    check("diff", calculus_tool("x**3"), "result", "3*x**2")
    check("integrate", calculus_tool("x**2", "integrate"), "result", "x**3/3")
    check("limit", calculus_tool("sin(x)/x", "limit", point="0"), "result", "1")
    check("summation", calculus_tool("x", "summation", lower="1", upper="100"), "result", "5050")


def test_matrix():
    print("\n--- matrix_tool ---")
    A = [[1, 2], [3, 4]]
    check("det", matrix_tool(A, "determinant"), "result", "-2")
    check("rank", matrix_tool(A, "rank"), "rank", 2)
    check("transpose", matrix_tool(A, "transpose"), "result", [["1","3"],["2","4"]])


def test_statistics():
    print("\n--- statistics_tool ---")
    data = [1, 2, 3, 4, 5]
    check("mean", statistics_tool(data, "mean"), "result", 3.0)
    check("median", statistics_tool(data, "median"), "result", 3)
    check("describe", statistics_tool(data, "describe"), "count", 5)


def test_code():
    print("\n--- code_tool ---")
    check("run", code_tool("print(2+2)", "run"), "stdout", "4\n")
    check("check ok", code_tool("x = 1", "check"), "valid", True)
    check("check bad", code_tool("def f(:", "check"), "valid", False)
    check("blocked", code_tool("import os", "check"), "valid", False)

if __name__ == "__main__":
    print("ReasonForge Sanity Checks...\n")

    test_math()
    test_calculus()
    test_matrix()
    test_statistics()
    test_code()

    print(f"\n{'-' * 48}")
    print(f"  {PASSED}/{TOTAL} passed")
    if PASSED == TOTAL:
        print("All checks passed!")
    else:
        print(f"{TOTAL - PASSED} check(s) FAILED.")
        sys.exit(1)
    print("=" * 48)
