"""Shared core — tool schemas, expert definitions, and LLM client."""

import inspect
import json
import typing
import requests

from experts.math.tools.algebra import math_tool, OPERATIONS as MATH_OPS, DOMAINS
from experts.math.tools.calculus import calculus_tool, OPERATIONS as CALC_OPS
from experts.math.tools.matrix import matrix_tool, OPERATIONS as MAT_OPS
from experts.math.tools.statistics import statistics_tool, OPERATIONS as STAT_OPS
from experts.code.tools.code import code_tool, OPERATIONS as CODE_OPS

DEFAULT_MODEL = "qwen3:32b"
DEFAULT_URL = "http://localhost:11434/api/chat"
MAX_ROUNDS = 5



_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation):
    """Map a Python type annotation to a JSON Schema type."""
    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        items = {"type": _json_type(args[0])} if args else {}
        return "array", items
    return _PY_TO_JSON.get(annotation, "string"), None


TOOL_ENUMS = {
    "math_tool": {"operation": sorted(MATH_OPS), "domain": sorted(DOMAINS.keys())},
    "calculus_tool": {"operation": sorted(CALC_OPS)},
    "matrix_tool": {"operation": sorted(MAT_OPS)},
    "statistics_tool": {"operation": sorted(STAT_OPS)},
    "code_tool": {"operation": sorted(CODE_OPS)},
}


def build_schema(fn, enums=None):
    """Build an OpenAI-format tool schema from a function's signature + docstring."""
    sig = inspect.signature(fn)
    props = {}
    required = []
    for name, param in sig.parameters.items():
        ann = param.annotation if param.annotation != inspect.Parameter.empty else str
        json_type, items = _json_type(ann)
        prop = {"type": json_type}
        if items:
            prop["items"] = items
        if enums and name in enums:
            prop["enum"] = enums[name]
        if param.default != inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        props[name] = prop

    doc = inspect.getdoc(fn) or fn.__name__
    desc = doc.split("\n\n", 1)[1].strip() if "\n\n" in doc else doc.split("\n")[0]

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }




_MATH_TOOLS = [math_tool, calculus_tool, matrix_tool, statistics_tool]
_CODE_TOOLS = [code_tool]

EXPERTS = {
    "Mathematician": {
        "system": (
            "You are a highly capable mathematician. "
            "You have access to specialized function tools. You MUST use the proper function calling API format to invoke them when you use tools. Be concise, no redundancy.\n"
            "**TRUST TOOL RESULTS BLINDLY. DO NOT RECOMPUTE OR QUESTION THEM.**"
        ),
        "tools": [
            build_schema(fn, TOOL_ENUMS.get(fn.__name__))
            for fn in _MATH_TOOLS
        ],
        "dispatch": {fn.__name__: fn for fn in _MATH_TOOLS},
    },
    "Coder": {
        "system": (
            "You are a precise coding assistant. "
            "You have access to code_tool — a sandboxed Python execution environment. "
            "ALWAYS use code_tool — NEVER guess output or run code in your head.\n"
            "Use code_tool with operation='check' to verify syntax before running.\n"
            "Use code_tool with operation='run' to execute and get actual output.\n"
            "Use code_tool with operation='ast_inspect' to analyze code structure.\n"
            "**TRUST TOOL RESULTS BLINDLY. DO NOT RECOMPUTE OR QUESTION THEM.**"
        ),
        "tools": [
            build_schema(fn, TOOL_ENUMS.get(fn.__name__))
            for fn in _CODE_TOOLS
        ],
        "dispatch": {fn.__name__: fn for fn in _CODE_TOOLS},
    },
}




def llm_request(messages, tools, model, url, stream=False, think=True):
    """Send a chat request to Ollama. Returns parsed JSON or raw response for streaming."""
    payload = {"model": model, "messages": messages, "stream": stream, "think": think}
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(url, json=payload, stream=stream, timeout=600)
        resp.raise_for_status()
        if not stream:
            return resp.json()
        return resp
    except requests.exceptions.HTTPError as e:
        body = e.response.text[:500]
        raise ConnectionError(f"HTTP {e.response.status_code} from {url}\n{body}")
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(f"Cannot connect to {url}. Is your model server running?\nDetail: {e}")
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Request to {url} timed out (600s).")


def iter_stream(resp):
    """Yield (token, thinking_token, tool_calls, done, msg) from a streaming Ollama response."""
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            yield (
                msg.get("content", ""),
                msg.get("thinking", ""),
                msg.get("tool_calls", []),
                chunk.get("done", False),
                msg,
            )
            if chunk.get("done"):
                break
    finally:
        resp.close()


def stream_to_msg(resp):
    """Consume a streaming Ollama response and return the final message dict."""
    content = ""
    tool_calls = []
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            content += msg.get("content", "")
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            if chunk.get("done"):
                break
    finally:
        resp.close()
    return {"role": "assistant", "content": content, "tool_calls": tool_calls}
