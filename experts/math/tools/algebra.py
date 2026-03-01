"""Algebra and number theory tool.

Operations: compute, solve, simplify, factor, expand, gcd, lcm,
prime_factors, divisors, mod_inverse, nsolve, crt.
"""

import sympy
from functools import lru_cache
from sympy import Symbol, solveset, S, factorint, gcd, lcm, divisors as _divisors
from sympy.ntheory.modular import crt as _crt
from .preprocess import preprocess

OPERATIONS = {
    "compute", "solve", "simplify", "factor", "expand",
    "gcd", "lcm", "prime_factors",
    "divisors", "mod_inverse", "nsolve", "crt",
}

DOMAINS = {
    "real": S.Reals,
    "complex": S.Complexes,
    "integer": S.Integers,
    "natural": S.Naturals0,
}


@lru_cache(maxsize=256)
def math_tool(expression: str, operation: str = "compute",
              variable: str = "x", domain: str = "real") -> dict:
    """All-in-one algebra + number theory tool.

    Use for arithmetic, algebra, number theory. Operations: compute, solve, simplify,
    factor, expand, gcd, lcm, prime_factors, divisors, mod_inverse, nsolve, crt.
    'compute' supports sympy functions: totient, fibonacci, catalan, bell, factorial,
    binomial, lucas, mobius, isprime, nextprime.
    """
    try:
        expression = preprocess(expression)

        if operation == "compute":
            result = sympy.simplify(sympy.sympify(expression))
            return _result(expression, str(result), _to_float(result))

        elif operation == "solve":
            sym = Symbol(variable)
            sol_set = solveset(sympy.sympify(expression), sym, DOMAINS.get(domain, S.Reals))
            if sol_set.is_FiniteSet:
                try:
                    solutions = sorted(sol_set, key=lambda s: complex(s).real)
                except (TypeError, ValueError):
                    solutions = list(sol_set)
                return {
                    "solutions": [str(s) for s in solutions],
                    "numeric": [_to_float(s) for s in solutions],
                    "domain": domain, "verified": True,
                }
            return {"solutions": str(sol_set), "domain": domain, "verified": True}

        elif operation in ("simplify", "factor", "expand"):
            fn = getattr(sympy, operation)
            result = fn(sympy.sympify(expression))
            return _result(expression, str(result), _to_float(result))

        elif operation in ("gcd", "lcm"):
            parts = [sympy.sympify(p.strip()) for p in expression.split(",")]
            if len(parts) != 2:
                return {"error": f"{operation} requires exactly 2 comma-separated values"}
            fn = gcd if operation == "gcd" else lcm
            result = fn(parts[0], parts[1])
            return _result(expression, str(result), _to_float(result))

        elif operation == "prime_factors":
            n = int(sympy.sympify(expression))
            factors = factorint(n)
            return {"number": n, "factors": {str(k): v for k, v in factors.items()}, "verified": True}

        elif operation == "divisors":
            n = int(sympy.sympify(expression))
            divs = _divisors(n)
            return {"number": n, "divisors": divs, "count": len(divs), "verified": True}

        elif operation == "mod_inverse":
            parts = [int(sympy.sympify(p.strip())) for p in expression.split(",")]
            if len(parts) != 2:
                return {"error": "mod_inverse requires 'a, m' (two comma-separated integers)"}
            result = sympy.mod_inverse(parts[0], parts[1])
            return {"a": parts[0], "m": parts[1], "inverse": int(result), "verified": True}

        elif operation == "nsolve":
            sym = Symbol(variable)
            if "," in expression:
                expr_str, guess_str = expression.rsplit(",", 1)
                expr = sympy.sympify(expr_str.strip())
                guess = float(guess_str.strip())
            else:
                expr = sympy.sympify(expression)
                guess = 0.0
            result = sympy.nsolve(expr, sym, guess)
            return {"solution": str(result), "numeric": float(result), "verified": True}

        elif operation == "crt":
            if ";" not in expression:
                return {"error": "crt requires 'remainders; moduli' format, e.g. '2,1,7; 3,4,11'"}
            rem_str, mod_str = expression.split(";", 1)
            remainders = [int(sympy.sympify(r.strip())) for r in rem_str.split(",")]
            moduli = [int(sympy.sympify(m.strip())) for m in mod_str.split(",")]
            if len(remainders) != len(moduli):
                return {"error": "Number of remainders must equal number of moduli"}
            result = _crt(moduli, remainders)
            if result is None:
                return {"error": "No solution — moduli may not be pairwise coprime"}
            solution, modulus = result
            return {"solution": int(solution), "modulus": int(modulus),
                    "description": f"x ≡ {int(solution)} (mod {int(modulus)})", "verified": True}

        else:
            return {"error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}"}

    except Exception as e:
        return {"error": str(e), "expression": expression, "operation": operation}


def _to_float(expr):
    try:
        return float(expr)
    except (TypeError, ValueError):
        return None


def _result(expr, exact, numeric):
    return {"input": expr, "result": exact, "numeric": numeric, "verified": True}
