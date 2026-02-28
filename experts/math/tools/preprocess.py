"""Expression preprocessor — uses SymPy's battle-tested parse_expr.

Handles:  ^ → **,  2x → 2*x,  (x+1)(x-1) → (x+1)*(x-1),  sin(x)cos(x) → sin(x)*cos(x)
Also:     infinity/inf → oo  (word-boundary safe)
All via SymPy's standard_transformations + implicit_multiplication + convert_xor.
"""

import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication,
    implicit_multiplication_application,
    convert_xor,
)

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication,
    convert_xor,
    implicit_multiplication_application,
)

# Word-boundary safe: matches standalone infinity/inf/+inf/-inf but NOT "information"
_INF_RE = re.compile(r'(?<![a-zA-Z])([+-]?\s*)(infinity|inf)(?![a-zA-Z])', re.IGNORECASE)


def _inf_replace(m):
    sign = m.group(1).replace(" ", "")
    return f"{sign}oo"


def preprocess(expression: str) -> str:
    """Convert natural math notation to SymPy-compatible syntax.

    Uses SymPy's own parser with implicit multiplication and ^ → ** conversion.
    Falls back to the original expression if parsing fails.
    A restricted local_dict blocks dangerous builtins (exec, eval, __import__, etc.).
    """
    try:
        s = expression.strip()

        # Replace standalone infinity/inf with oo (word-boundary safe)
        s = _INF_RE.sub(_inf_replace, s)

        # Comma-separated inputs (e.g. "12, 18" for GCD/LCM) — just fix ^ and return
        if "," in s:
            return s.replace("^", "**")

        # Restricted namespace: block dangerous builtins while allowing SymPy symbols
        _blocked = {name: None for name in (
            "exec", "eval", "__import__", "open", "compile",
            "globals", "locals", "getattr", "setattr", "delattr",
            "breakpoint", "exit", "quit", "input", "print",
        )}
        parsed = parse_expr(s, local_dict=_blocked,
                            transformations=_TRANSFORMATIONS, evaluate=False)
        return str(parsed)
    except Exception:
        # If SymPy can't parse it, return as-is and let the downstream tool handle the error
        return expression.strip()

