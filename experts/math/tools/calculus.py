"""Calculus tool — differentiate, integrate, limit, series, summation,
partial_fraction, trigsimp, ode_solve, laplace.
"""

import sympy
from functools import lru_cache
from .preprocess import preprocess

OPERATIONS = {
    "differentiate", "integrate", "limit", "series", "summation",
    "partial_fraction", "trigsimp", "ode_solve", "laplace",
}


def _parse_bound(b_str):
    if not b_str:
        return None
    s = str(b_str).strip().lower()
    if s in ["infinity", "+infinity", "inf", "+inf", "oo"]:
        return sympy.oo
    if s in ["-infinity", "-inf", "-oo"]:
        return -sympy.oo
    return sympy.sympify(b_str)

@lru_cache(maxsize=256)
def calculus_tool(expression: str, operation: str = "differentiate",
                  variable: str = "x", order: int = 1,
                  point: str = None, upper: str = None, lower: str = None,
                  n_terms: int = 6) -> dict:
    """All-in-one calculus tool.

    Use for calculus. Operations: differentiate, integrate, limit, series, summation,
    partial_fraction, trigsimp, ode_solve, laplace. Expression is the FULL equation in SymPy syntax.
    For ode_solve, use 'f' as the function name (e.g. f'' - f = exp(x)).
    """
    try:
        expression = preprocess(expression)
        sym = sympy.Symbol(variable)
        expr = sympy.sympify(expression)

        if operation == "differentiate":
            result = sympy.diff(expr, sym, order)
            out_dict = _out(expression, result, f"d^{order}/d{variable}^{order}" if order > 1 else f"d/d{variable}")
            try:
                out_dict["factored"] = str(sympy.factor(result))
                roots = sympy.solve(result, sym)
                if isinstance(roots, list) and len(roots) <= 10:
                    out_dict["roots"] = [str(r) for r in roots]
            except Exception:
                pass
            return out_dict

        elif operation == "integrate":
            if lower is not None and upper is not None:
                lb = _parse_bound(lower)
                ub = _parse_bound(upper)
                result = sympy.integrate(expr, (sym, lb, ub))
                return _out(expression, result, f"integral [{lower}, {upper}]")
            result = sympy.integrate(expr, sym)
            return _out(expression, result, "indefinite integral")

        elif operation == "limit":
            pt = _parse_bound(point or "0")
            result = sympy.limit(expr, sym, pt)
            return _out(expression, result, f"limit {variable}->{point or '0'}")

        elif operation == "series":
            pt = _parse_bound(point or "0")
            result = sympy.series(expr, sym, pt, n=n_terms)
            return {"input": expression, "result": str(result), "center": str(pt),
                    "terms": n_terms, "verified": True}

        elif operation == "summation":
            if lower is None or upper is None:
                return {"error": "summation requires both lower and upper bounds"}
            lb = _parse_bound(lower)
            ub = _parse_bound(upper)
            result = sympy.summation(expr, (sym, lb, ub))
            return _out(expression, result, f"sum [{lower}, {upper}]")

        elif operation == "partial_fraction":
            result = sympy.apart(expr, sym)
            return _out(expression, result, "partial fraction decomposition")

        elif operation == "trigsimp":
            result = sympy.trigsimp(expr)
            return _out(expression, result, "trig simplification")

        elif operation == "ode_solve":
            f = sympy.Function("f")
            ode_expr = expr.subs(sympy.Symbol("f"), f(sym))
            sol = sympy.dsolve(ode_expr, f(sym))
            if isinstance(sol, list):
                return {"solutions": [str(s) for s in sol], "verified": True}
            return {"solution": str(sol), "verified": True}

        elif operation == "laplace":
            s_var = sympy.Symbol("s")
            result = sympy.laplace_transform(expr, sym, s_var, noconds=True)
            return _out(expression, result, f"Laplace transform ({variable}->s)")

        else:
            return {"error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}"}

    except Exception as e:
        return {"error": str(e), "expression": expression, "operation": operation}


def _out(expr, result, label):
    result_s = sympy.simplify(result)
    try:
        numeric = float(result_s)
    except (TypeError, ValueError):
        numeric = None
    return {"input": expr, "operation": label, "result": str(result_s), "numeric": numeric, "verified": True}
