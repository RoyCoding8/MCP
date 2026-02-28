"""Unit tests for the Math Expert tools.

Run:  uv run python -m tests.test_math_tools
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from experts.math.tools.algebra import math_tool
from experts.math.tools.calculus import calculus_tool
from experts.math.tools.matrix import matrix_tool
from experts.math.tools.statistics import statistics_tool

TOTAL = 0
PASSED = 0


def check(label, result, key, expected):
    global TOTAL, PASSED
    TOTAL += 1
    actual = result.get(key)
    # normalize list comparisons (sort numerically)
    def norm(v):
        if isinstance(v, list):
            try: return sorted(v, key=lambda x: float(x) if x is not None else 0)
            except: return sorted(str(x) for x in v)
        return v
    ok = norm(actual) == norm(expected)
    PASSED += ok
    print(f"  {'[PASS]' if ok else '[FAIL]'}  {label}")
    if not ok:
        print(f"         Expected {key}={expected!r}")
        print(f"         Got      {key}={actual!r}")
    return ok


# ── Algebra ────────────────────────────────────────────

def test_algebra():
    print("\n--- math_tool (algebra) ----------------------------------")
    check("compute 347*892",    math_tool("347*892"),                    "result", "309524")
    check("compute sqrt(144)",  math_tool("sqrt(144)"),                  "result", "12")
    check("compute fraction",   math_tool("3/7 + 2/5"),                  "result", "29/35")
    check("solve quadratic",    math_tool("x**2-3*x+2", "solve"),        "solutions", ["1", "2"])
    check("solve cubic",        math_tool("x**3-6*x**2+11*x-6", "solve"), "solutions", ["1", "2", "3"])
    check("solve no real",      math_tool("x**2+1", "solve", domain="real"), "solutions", [])
    check("simplify",           math_tool("(x**2-1)/(x-1)", "simplify"), "result", "x + 1")
    check("factor",             math_tool("x**2-5*x+6", "factor"),       "result", "(x - 3)*(x - 2)")
    check("expand",             math_tool("(x+1)*(x+2)", "expand"),      "result", "x**2 + 3*x + 2")
    check("gcd",                math_tool("12, 18", "gcd"),              "result", "6")
    check("lcm",                math_tool("4, 6", "lcm"),               "result", "12")
    check("prime_factors",      math_tool("360", "prime_factors"),       "factors", {"2": 3, "3": 2, "5": 1})


# ── Calculus ───────────────────────────────────────────

def test_calculus():
    print("\n--- calculus_tool ----------------------------------------")
    check("diff x^3",           calculus_tool("x**3"),                              "result", "3*x**2")
    check("diff sin(x)",        calculus_tool("sin(x)"),                            "result", "cos(x)")
    check("2nd derivative",     calculus_tool("x**4", order=2),                     "result", "12*x**2")
    check("integrate x^2",      calculus_tool("x**2", "integrate"),                 "result", "x**3/3")
    check("definite integral",  calculus_tool("x**2", "integrate", lower="0", upper="1"), "result", "1/3")
    check("limit 1/x -> oo",    calculus_tool("1/x", "limit", point="oo"),          "result", "0")
    check("limit sin(x)/x",     calculus_tool("sin(x)/x", "limit", point="0"),      "result", "1")
    check("summation 1..100",   calculus_tool("x", "summation", lower="1", upper="100"), "result", "5050")


# ── Matrix ─────────────────────────────────────────────

def test_matrix():
    print("\n--- matrix_tool ------------------------------------------")
    A = [[1, 2], [3, 4]]
    check("det 2x2",            matrix_tool(A, "determinant"),              "result", "-2")
    check("rank 2x2",           matrix_tool(A, "rank"),                     "rank", 2)
    check("transpose",          matrix_tool(A, "transpose"),                "result", [["1","3"],["2","4"]])
    check("eigenvalues",        matrix_tool([[2,0],[0,3]], "eigenvalues"),   "eigenvalues", {"2": 1, "3": 1})

    I = [[1,0],[0,1]]
    check("multiply A*I",       matrix_tool(A, "multiply", matrix_b=I),     "result", [["1","2"],["3","4"]])

    check("rref",               matrix_tool([[1,2,3],[4,5,6]], "rref"),      "pivots", [0, 1])


# ── Statistics ─────────────────────────────────────────

def test_statistics():
    print("\n--- statistics_tool --------------------------------------")
    data = [1, 2, 3, 4, 5]
    check("mean",               statistics_tool(data, "mean"),                  "result", 3.0)
    check("median",             statistics_tool(data, "median"),                 "result", 3)
    r = statistics_tool(data, "std")
    check("std",                r,                                              "result", r["result"] if abs(r["result"] - 1.5811388300841898) < 1e-6 else 1.5811388300841898)
    check("describe count",     statistics_tool(data, "describe"),               "count", 5)

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    check("correlation",        statistics_tool(x, "correlation", data_y=y),     "correlation", 1.0)
    check("regression slope",   statistics_tool(x, "regression", data_y=y),      "slope", 2.0)
    check("regression r^2",     statistics_tool(x, "regression", data_y=y),      "r_squared", 1.0)

    check("percentile 50",      statistics_tool(data, "percentile", percentile_value=50), "result", 3)


# ── Main ───────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("  ReasonForge Phase 2 -- Math Expert Tests")
    print("=" * 52)

    test_algebra()
    test_calculus()
    test_matrix()
    test_statistics()

    print("\n" + "=" * 52)
    print(f"  {PASSED}/{TOTAL} passed")
    if PASSED == TOTAL:
        print("  All tests passed!")
    else:
        print(f"  {TOTAL - PASSED} test(s) FAILED.")
    print("=" * 52)
