"""ReasonForge Chat UI -- Google-inspired Gradio interface with expert selection.

Run:  uv run python -m ui.app
"""

import inspect
import json
import pathlib
import re
import typing
import gradio as gr
from experts.math.tools.algebra import math_tool, OPERATIONS as MATH_OPS, DOMAINS
from experts.math.tools.calculus import calculus_tool, OPERATIONS as CALC_OPS
from experts.math.tools.matrix import matrix_tool, OPERATIONS as MAT_OPS
from experts.math.tools.statistics import statistics_tool, OPERATIONS as STAT_OPS
from experts.code.tools.code import code_tool, OPERATIONS as CODE_OPS

# -- Config --

DEFAULT_MODEL = "qwen3:32b"
DEFAULT_URL = "http://localhost:11434/api/chat"
MAX_ROUNDS = 5

# -- Auto-schema generation --

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation):
    """Map a Python type annotation to a JSON Schema type."""
    origin = typing.get_origin(annotation)
    if origin is list or origin is typing.List:
        args = typing.get_args(annotation)
        items = {"type": _json_type(args[0])} if args else {}
        return "array", items
    return _PY_TO_JSON.get(annotation, "string"), None


# Enum overrides for operation/domain params (read from the tools themselves)
TOOL_ENUMS = {
    "math_tool": {"operation": sorted(MATH_OPS), "domain": sorted(DOMAINS.keys())},
    "calculus_tool": {"operation": sorted(CALC_OPS)},
    "matrix_tool": {"operation": sorted(MAT_OPS)},
    "statistics_tool": {"operation": sorted(STAT_OPS)},
    "code_tool": {"operation": sorted(CODE_OPS)},
}


def _build_schema(fn, enums=None):
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

    # Extract first paragraph of docstring as description
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


# -- Expert definitions --

_MATH_TOOLS = [math_tool, calculus_tool, matrix_tool, statistics_tool]
_CODE_TOOLS = [code_tool]

EXPERTS = {
    "Mathematician": {
        "system": (
            "You are a highly capable mathematician. "
            "You have access to specialized function tools. You MUST use the proper function calling API format to invoke them when you use tools. Be concise, no redundancy.\n"
            "**NEVER QUESTION THE TOOL RESULTS TO INPUTS. TRUST IT BLINDLY. QUESTIONING TOOL WASTES TIME!**"
        ),
        "tools": [
            _build_schema(fn, TOOL_ENUMS.get(fn.__name__))
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
            "**TRUST TOOL RESULTS. DO NOT RECOMPUTE OR QUESTION THEM.**"
        ),
        "tools": [
            _build_schema(fn, TOOL_ENUMS.get(fn.__name__))
            for fn in _CODE_TOOLS
        ],
        "dispatch": {fn.__name__: fn for fn in _CODE_TOOLS},
    },
}


# -- LLM Client --

def _llm_request(messages, tools, model, url, stream=False):
    """Send a chat request to Ollama. Returns parsed JSON or raw response for streaming."""
    import urllib.request, urllib.error

    payload = {"model": model, "messages": messages, "stream": stream}
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

    try:
        resp = urllib.request.urlopen(req, timeout=300)
        if not stream:
            return json.loads(resp.read())
        return resp
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise ConnectionError(f"HTTP {e.code} from {url}\n{body}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot connect to {url}. Is your model server running?\nDetail: {e}")
    except TimeoutError:
        raise ConnectionError(f"Request to {url} timed out (300s).")


def _iter_stream(resp):
    """Yield (token, tool_calls, done, msg) from a streaming Ollama response."""
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        chunk = json.loads(line)
        msg = chunk.get("message", {})
        yield msg.get("content", ""), msg.get("tool_calls", []), chunk.get("done", False), msg
        if chunk.get("done"):
            break


# -- Chat logic --

def _chat_list(history, message, accumulated):
    """Build a Gradio-compatible chat list with user + assistant messages."""
    return history + [
        {"role": "user", "content": _to_str(message)},
        {"role": "assistant", "content": accumulated},
    ]
LOADING_DOTS = '<span class="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></span>'


def respond(message, history, expert_name, model_name, endpoint_url):
    """Two-phase chat handler: Compute → Present.

    Phase 1 (COMPUTE): /no_think, tools ON.  Model calls tools.
    Phase 2 (PRESENT): Thinking ON, no tools. Model reasons + writes answer.
    """
    expert = EXPERTS.get(expert_name)
    if not expert:
        yield _chat_list(history, message, "Expert not found.")
        return

    user_content = _to_str(message)
    base_system = expert["system"]

    compute_system = (
        base_system + "\n\n"
        "You are an expert reasoning agent. You solve problems by thinking step-by-step and using tools.\n"
        "You may use the <think>...</think> tags to reason before deciding what to do.\n"
        "If you need to compute or look up something, output the proper JSON tool call. I will execute it and return the result.\n"
        "If you have the final answer, simply present it to the user. Do not call tools if you already have the answer."
    )
    
    messages = [{"role": "system", "content": compute_system}]
    for h in history:
        messages.append({"role": h["role"], "content": _to_str(h.get("content", ""))})
    messages.append({"role": "user", "content": user_content})

    accumulated = ""

    for round_num in range(MAX_ROUNDS):
        # We start streaming the model's thought process / answer immediately
        yield _chat_list(history, message, accumulated + "\n\n" + LOADING_DOTS)

        try:
            resp = _llm_request(messages, expert["tools"], model_name, endpoint_url, stream=True)
        except Exception as e:
            accumulated += f"\n\n**Error:** {e}"
            yield _chat_list(history, message, accumulated)
            return

        full_content = ""
        all_tool_calls = []
        last_msg = {}

        try:
            for token, tool_calls, done, msg_chunk in _iter_stream(resp):
                if tool_calls:
                    all_tool_calls.extend(tool_calls)
                if token:
                    full_content += token
                    yield _chat_list(history, message, accumulated + "\n\n" + _clean_response(full_content))
                last_msg = msg_chunk
        finally:
            resp.close()

        accumulated += "\n\n" + _clean_response(full_content)
        yield _chat_list(history, message, accumulated)

        # If the model didn't call any tools, it's done reasoning and has provided the final answer!
        if not all_tool_calls:
            break

        # Otherwise, process the tool calls
        messages.append({
            "role": last_msg.get("role", "assistant"),
            "content": full_content,
            "tool_calls": all_tool_calls,
        })

        for tc in all_tool_calls:
            name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            brief = ", ".join(f"{k}={_short(v)}" for k, v in args.items())

            if name in expert["dispatch"]:
                try:
                    result = expert["dispatch"][name](**args)
                    if isinstance(result, dict) and "error" in result:
                        result["hint"] = f"Check arguments. Available tools: {', '.join(expert['dispatch'].keys())}"
                except Exception as e:
                    result = {
                        "error": f"{name} failed: {e}",
                        "hint": f"Check arguments. Available tools: {', '.join(expert['dispatch'].keys())}"
                    }
            else:
                result = {
                    "error": f"Unknown tool: {name}",
                    "available_tools": list(expert["dispatch"].keys())
                }

            result_str = json.dumps(result)
            display = json.dumps(result, indent=2)
            if len(display) > 300:
                display = display[:300] + "\n..."

            accumulated += (
                f'\n\n<details><summary><code>{name}({brief})</code></summary>'
                f'\n\n```json\n{display}\n```\n\n</details>\n'
            )
            yield _chat_list(history, message, accumulated)
            messages.append({"role": "tool", "content": result_str})


# -- Helpers --

def _short(v, limit=60):
    s = str(v)
    return s if len(s) <= limit else s[:limit] + "..."


def _to_str(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", item.get("content", str(item))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _clean_response(text):
    """Collapse <think> blocks into styled, collapsible details.
    Forces the block open while streaming, then closes it when complete.
    """
    def _replace_closed(m):
        thought = m.group(1).strip()
        if not thought:
            return ""
        return (
            '\n<details class="thinking"><summary>Thinking...</summary>\n'
            f'\n{thought}\n</details>\n\n'
        )

    def _replace_open(m):
        thought = m.group(1).strip()
        if not thought:
            return ""
        # The 'open' attribute keeps it expanded while streaming.
        # Gradio redraws the DOM on every token, so manual clicks would be overwritten.
        return (
            '\n<details class="thinking" open><summary>Thinking...</summary>\n'
            f'\n{thought}\n</details>\n\n'
        )

    text = re.sub(r'<think>(.*?)</think>', _replace_closed, text, flags=re.DOTALL)
    text = re.sub(r'<think>(.*?)$', _replace_open, text, flags=re.DOTALL)
    return text.strip()


# -- CSS --

_CSS_PATH = str(pathlib.Path(__file__).with_name("style.css"))



# -- UI --

def build_ui():
    with gr.Blocks(title="ReasonForge") as app:

        # Header row with title + dark mode toggle
        with gr.Row(elem_id="rf-header-row"):
            gr.Markdown(
                "# ReasonForge\nDeterministic tools for small language models — choose your expert.",
                elem_id="rf-header",
            )
            dark_btn = gr.Button("🌙", elem_id="rf-theme-toggle", scale=0, min_width=42)

        # Dark mode toggle via JS
        dark_btn.click(
            fn=None, inputs=None, outputs=None,
            js="""() => {
                document.querySelector('body').classList.toggle('dark');
                const btn = document.querySelector('#rf-theme-toggle');
                btn.textContent = document.querySelector('body').classList.contains('dark') ? '☀️' : '🌙';
            }"""
        )

        with gr.Accordion("⚙ Settings", open=False):
            with gr.Row():
                expert_dd = gr.Dropdown(choices=list(EXPERTS.keys()), value="Mathematician", label="Expert", scale=1)
                model_tb = gr.Textbox(value=DEFAULT_MODEL, label="Model", scale=1)
                url_tb = gr.Textbox(value=DEFAULT_URL, label="Endpoint", scale=2)

        chatbot = gr.Chatbot(
            show_label=False,
            sanitize_html=False,
            latex_delimiters=[
                {"left": "$$", "right": "$$", "display": True},
                {"left": "$", "right": "$", "display": False},
                {"left": "\\(", "right": "\\)", "display": False},
                {"left": "\\[", "right": "\\]", "display": True},
            ],
            elem_id="rf-chatbot",
        )

        with gr.Row(elem_id="rf-input-pill"):
            msg = gr.Textbox(placeholder="Ask a math question...", show_label=False, scale=1, container=False, elem_id="rf-msg-box")
            with gr.Column(elem_id="rf-btn-container", scale=0, min_width=36):
                send_btn = gr.Button("↑", variant="primary", elem_id="rf-send-btn", elem_classes=["rf-action-btn", "rf-inactive"])
                stop_btn = gr.Button("■", variant="primary", visible=False, elem_id="rf-stop-btn", elem_classes=["rf-action-btn"])

        gr.Examples(
            examples=[
                "Evaluate the limit of $(1 + 1/x)^x$ as $x \\to \\infty$",
                "Compute the matrix eigenvalues for [[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
                "Find the integral of $e^{-x^2}$ from $-\\infty$ to $\\infty$",
                "What is the sum of $1/n^2$ from $n=1$ to $\\infty$?",
                "Solve the differential equation $f''(x) - f(x) = e^x$",
            ],
            inputs=msg,
        )

        # Cosmetic only: grey out button if empty
        def toggle_play(text):
            is_valid = bool(text and text.strip())
            return gr.update(elem_classes=["rf-action-btn", "rf-active" if is_valid else "rf-inactive"])

        msg.change(toggle_play, [msg], [send_btn], queue=False)

        def user_submit(message, chat_history):
            if not message or not message.strip():
                # Abort chain if empty
                raise gr.Error("Please enter a message.")
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            return "", chat_history

        def bot_respond(chat_history, expert_name, model_name, endpoint_url):
            if not chat_history:
                return chat_history
            user_msg = chat_history[-1]["content"]
            prev = chat_history[:-1]
            try:
                for updated in respond(user_msg, prev, expert_name, model_name, endpoint_url):
                    yield updated
            except Exception as e:
                # Catch any generator exceptions so the UI doesn't silently freeze
                chat_history.append({"role": "assistant", "content": f"**Fatal Error:** {e}"})
                yield chat_history

        # --- Generation state ---
        gen_state = gr.State(False)  # True while generating

        def pre_gen():
            """Hide send, show stop."""
            return gr.update(visible=False), gr.update(visible=True)

        def post_gen(text):
            """Hide stop, show send."""
            is_valid = bool(text and text.strip())
            return (
                gr.update(visible=True, elem_classes=["rf-action-btn", "rf-active" if is_valid else "rf-inactive"]),
                gr.update(visible=False),
            )

        # --- Submission chains ---

        # 1. Textbox Enter
        user_sub_ev = msg.submit(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False
        )
        pre_gen_ev = user_sub_ev.then(
            pre_gen, None, [send_btn, stop_btn], queue=False
        )
        bot_res_ev = pre_gen_ev.then(
            bot_respond, [chatbot, expert_dd, model_tb, url_tb], chatbot
        )
        bot_res_ev.then(
            post_gen, msg, [send_btn, stop_btn], queue=False
        )

        # 2. Send Button Click
        btn_sub_ev = send_btn.click(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False
        )
        btn_pre_ev = btn_sub_ev.then(
            pre_gen, None, [send_btn, stop_btn], queue=False
        )
        btn_res_ev = btn_pre_ev.then(
            bot_respond, [chatbot, expert_dd, model_tb, url_tb], chatbot
        )
        btn_res_ev.then(
            post_gen, msg, [send_btn, stop_btn], queue=False
        )

        # 3. Stop Button — cancel generation AND restore buttons in one handler
        stop_btn.click(
            fn=post_gen,
            inputs=msg,
            outputs=[send_btn, stop_btn],
            cancels=[bot_res_ev, btn_res_ev],
            queue=False,
        )

    return app


if __name__ == "__main__":
    import os
    app = build_ui()
    share = os.environ.get("RF_SHARE", "").lower() in ("1", "true", "yes")

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.blue,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=share,
        css_paths=[_CSS_PATH],
        theme=theme
    )
