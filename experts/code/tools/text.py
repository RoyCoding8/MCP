"""Text tool — deterministic string and regex operations.

Operations: regex_find, regex_replace, count, split, json_parse, diff.
Pure Python — no subprocess, no external deps.
"""

import difflib
import json
import re

OPERATIONS = {"regex_find", "regex_replace", "count", "split", "json_parse", "diff"}

_MAX_OUTPUT = 2000  # Max chars of results to return


def text_tool(text: str, operation: str = "regex_find",
              pattern: str = "", replacement: str = "",
              text_b: str = "") -> dict:
    """Deterministic string and regex operations.

    Use for regex matching, text replacement, string splitting, JSON parsing,
    and computing diffs. Operations: regex_find, regex_replace, count, split,
    json_parse, diff. Small models should delegate all regex work to this tool.
    """
    try:
        if operation == "regex_find":
            if not pattern:
                return {"error": "pattern is required for regex_find"}
            try:
                matches = re.findall(pattern, text)
            except re.error as e:
                return {"error": f"Invalid regex: {e}"}
            return {"matches": matches[:100], "count": len(matches), "verified": True}

        elif operation == "regex_replace":
            if not pattern:
                return {"error": "pattern is required for regex_replace"}
            try:
                result, count = re.subn(pattern, replacement, text)
            except re.error as e:
                return {"error": f"Invalid regex: {e}"}
            result_trunc = result[:_MAX_OUTPUT]
            out = {"result": result_trunc, "replacements": count, "verified": True}
            if len(result) > _MAX_OUTPUT:
                out["truncated"] = True
            return out

        elif operation == "count":
            if pattern:
                # Count regex matches
                try:
                    count = len(re.findall(pattern, text))
                except re.error as e:
                    return {"error": f"Invalid regex: {e}"}
                return {"count": count, "pattern": pattern, "verified": True}
            else:
                # Character/line/word counts
                return {
                    "characters": len(text),
                    "lines": text.count("\n") + 1 if text else 0,
                    "words": len(text.split()),
                    "verified": True,
                }

        elif operation == "split":
            if pattern:
                try:
                    parts = re.split(pattern, text)
                except re.error as e:
                    return {"error": f"Invalid regex: {e}"}
            else:
                parts = text.splitlines()
            return {"parts": parts[:200], "count": len(parts), "verified": True}

        elif operation == "json_parse":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {e.msg}", "line": e.lineno, "column": e.colno, "verified": True}

            if isinstance(parsed, dict):
                return {"type": "object", "keys": list(parsed.keys()), "size": len(parsed), "verified": True}
            elif isinstance(parsed, list):
                return {"type": "array", "length": len(parsed), "verified": True}
            else:
                return {"type": type(parsed).__name__, "value": str(parsed)[:500], "verified": True}

        elif operation == "diff":
            if not text_b:
                return {"error": "text_b is required for diff"}
            a_lines = text.splitlines(keepends=True)
            b_lines = text_b.splitlines(keepends=True)
            diff = list(difflib.unified_diff(a_lines, b_lines, fromfile="a", tofile="b"))
            diff_str = "".join(diff)[:_MAX_OUTPUT]
            added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
            out = {"diff": diff_str, "added": added, "removed": removed, "verified": True}
            if len("".join(diff)) > _MAX_OUTPUT:
                out["truncated"] = True
            return out

        else:
            return {"error": f"Unknown operation '{operation}'. Use: {', '.join(sorted(OPERATIONS))}"}

    except Exception as e:
        return {"error": str(e), "operation": operation}
