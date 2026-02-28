"""Matrix tool — determinant, inverse, eigenvalues, eigenvectors, rank, rref,
transpose, multiply, add, trace, nullspace, columnspace, charpoly, norm, adjugate, solve.
"""

import sympy
from sympy import Matrix
from sympy.parsing.sympy_parser import parse_expr

OPERATIONS = {
    "determinant", "inverse", "eigenvalues", "eigenvectors",
    "rank", "rref", "transpose", "multiply", "add",
    "trace", "nullspace", "columnspace", "charpoly", "norm", "adjugate", "solve",
}


def matrix_tool(matrix: list[list], operation: str = "determinant",
                matrix_b: list[list] = None) -> dict:
    """All-in-one matrix tool.

    Use for matrix math. Operations: determinant, inverse, eigenvalues, eigenvectors,
    rank, rref, transpose, multiply, add, trace, nullspace, columnspace, charpoly, norm,
    adjugate, solve (Ax=b). Pass matrix as [[row1],[row2]].
    """
    try:
        A = _parse_matrix(matrix)

        if operation == "determinant":
            return {"result": str(A.det()), "verified": True}

        elif operation == "inverse":
            return {"result": _mat_to_list(A.inv()), "verified": True}

        elif operation == "eigenvalues":
            evals = A.eigenvals()
            return {"eigenvalues": {str(k): v for k, v in evals.items()}, "verified": True}

        elif operation == "eigenvectors":
            evecs = A.eigenvects()
            out = []
            for val, mult, vecs in evecs:
                out.append({
                    "eigenvalue": str(val),
                    "multiplicity": mult,
                    "vectors": [_mat_to_list(v) for v in vecs],
                })
            return {"eigenvectors": out, "verified": True}

        elif operation == "rank":
            return {"rank": A.rank(), "shape": list(A.shape), "verified": True}

        elif operation == "rref":
            rref_mat, pivots = A.rref()
            return {"rref": _mat_to_list(rref_mat), "pivots": list(pivots), "verified": True}

        elif operation == "transpose":
            return {"result": _mat_to_list(A.T), "verified": True}

        elif operation in ("multiply", "add"):
            if matrix_b is None:
                return {"error": f"{operation} requires matrix_b"}
            B = _parse_matrix(matrix_b)
            result = A * B if operation == "multiply" else A + B
            return {"result": _mat_to_list(result), "verified": True}

        elif operation == "trace":
            return {"result": str(A.trace()), "verified": True}

        elif operation == "nullspace":
            ns = A.nullspace()
            return {"nullspace": [_mat_to_list(v) for v in ns], "dimension": len(ns), "verified": True}

        elif operation == "columnspace":
            cs = A.columnspace()
            return {"columnspace": [_mat_to_list(v) for v in cs], "dimension": len(cs), "verified": True}

        elif operation == "charpoly":
            lam = sympy.Symbol("lambda")
            cp = A.charpoly(lam)
            return {"charpoly": str(cp.as_expr()), "verified": True}

        elif operation == "norm":
            return {"result": str(A.norm()), "verified": True}

        elif operation == "adjugate":
            return {"result": _mat_to_list(A.adjugate()), "verified": True}

        elif operation == "solve":
            if matrix_b is None:
                return {"error": "solve (Ax=b) requires matrix_b as the b vector/matrix"}
            b = _parse_matrix(matrix_b)
            x = A.solve(b)
            return {"result": _mat_to_list(x), "verified": True}

        else:
            return {"error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}"}

    except Exception as e:
        return {"error": str(e), "operation": operation}


def _parse_matrix(data):
    rows = []
    for row in data:
        rows.append([parse_expr(str(x)) if isinstance(x, str) else x for x in row])
    return Matrix(rows)


def _mat_to_list(m):
    return [[str(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]
