"""Expression preprocessor — converts natural math notation to SymPy syntax.

Handles: ^ → **, 2x → 2*x, (x+1)(x-1) → (x+1)*(x-1), infinity/inf → oo.
Uses SymPy's parse_expr with implicit_multiplication + convert_xor.
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

_INF_RE = re.compile(r'(?<![a-zA-Z])([+-]?\s*)(infinity|inf)(?![a-zA-Z])', re.IGNORECASE)

_BLOCKED = {name: None for name in (
    "exec", "eval", "__import__", "open", "compile",
    "globals", "locals", "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit", "input", "print",
)}


def _inf_replace(m):
    sign = m.group(1).replace(" ", "")
    return f"{sign}oo"


def preprocess(expression: str) -> str:
    """Convert natural math notation to SymPy-compatible syntax.

    Falls back to the original expression if parsing fails.
    """
    try:
        s = expression.strip()
        s = _INF_RE.sub(_inf_replace, s)

        if "," in s:
            parts = []
            for part in s.split(","):
                try:
                    parsed = parse_expr(part.strip(), local_dict=_BLOCKED,
                                        transformations=_TRANSFORMATIONS, evaluate=False)
                    parts.append(str(parsed))
                except Exception:
                    parts.append(part.strip().replace("^", "**"))
            return ", ".join(parts)

        parsed = parse_expr(s, local_dict=_BLOCKED,
                            transformations=_TRANSFORMATIONS, evaluate=False)
        return str(parsed)
    except Exception:
        return expression.strip()
