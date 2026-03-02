"""ReasonForge Chat UI — Gradio interface with expert selection and two-phase pipeline."""

import json
import pathlib
import re
import gradio as gr
from core import (
    DEFAULT_MODEL, DEFAULT_URL, MAX_ROUNDS,
    EXPERTS,
    llm_request as _llm_request,
    iter_stream as _iter_stream,
)

LOADING_DOTS = '<span class="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></span>'

_CSS_PATH = str(pathlib.Path(__file__).with_name("style.css"))


def _chat_list(history, message, accumulated):
    """Build a Gradio-compatible chat list with user + assistant messages."""
    return history + [
        {"role": "user", "content": _to_str(message)},
        {"role": "assistant", "content": accumulated},
    ]


def respond(message, history, expert_name, model_name, endpoint_url):
    """Two-phase chat: Compute (tools ON, thinking OFF) then Present (thinking ON, no tools)."""
    expert = EXPERTS.get(expert_name)
    if not expert:
        yield _chat_list(history, message, "Expert not found.")
        return

    user_content = _to_str(message)

    compute_system = (
        expert["system"] + "\n\n"
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
        yield _chat_list(history, message, accumulated + "\n\n" + LOADING_DOTS)

        try:
            resp = _llm_request(messages, expert["tools"], model_name, endpoint_url, stream=True)
        except Exception as e:
            accumulated += f"\n\n**Error:** {e}"
            yield _chat_list(history, message, accumulated)
            return

        full_content = ""
        full_thinking = ""
        all_tool_calls = []
        last_msg = {}

        try:
            for token, thinking_token, tool_calls, done, msg_chunk in _iter_stream(resp):
                if tool_calls:
                    all_tool_calls.extend(tool_calls)
                changed = False
                if thinking_token:
                    full_thinking += thinking_token
                    changed = True
                if token:
                    full_content += token
                    changed = True
                if changed:
                    display = ""
                    if full_thinking:
                        display += "<think>" + full_thinking
                        if full_content:
                            display += "</think>"
                    display += full_content
                    yield _chat_list(history, message, accumulated + "\n\n" + _clean_response(display))
                last_msg = msg_chunk
        finally:
            resp.close()

        final_display = ""
        if full_thinking:
            final_display += "<think>" + full_thinking + "</think>"
        final_display += full_content

        accumulated += "\n\n" + _clean_response(final_display)
        yield _chat_list(history, message, accumulated)

        if not all_tool_calls:
            break

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
    """Collapse <think> blocks into styled, collapsible details."""
    def _replace_closed(m):
        thought = m.group(1).strip()
        if not thought:
            return ""
        lines = len(thought.split('\n'))
        label = f"Thought process ({lines} lines)"
        return (
            f'\n<details class="thinking thinking-done"><summary>'
            f'{label}</summary>'
            f'\n<div class="think-body">\n\n{thought}\n\n</div></details>\n\n'
        )

    def _replace_open(m):
        thought = m.group(1).strip()
        if not thought:
            return ""
        return (
            '\n<div class="thinking thinking-live">'
            '<div class="think-header">'
            '<span class="think-label">Thinking</span>'
            '<span class="think-dots"><span>.</span><span>.</span><span>.</span></span>'
            '</div>'
            f'\n<div class="think-body">\n\n{thought}\n\n</div></div>\n\n'
        )

    text = re.sub(r'<think>(.*?)</think>', _replace_closed, text, flags=re.DOTALL)
    text = re.sub(r'<think>(.*?)$', _replace_open, text, flags=re.DOTALL)
    return text.strip()




def build_ui():
    with gr.Blocks(title="ReasonForge") as app:

        with gr.Row(elem_id="rf-header-row"):
            gr.Markdown(
                "# ReasonForge\nDeterministic tools for small language models — choose your expert.",
                elem_id="rf-header",
            )
            dark_btn = gr.Button("🌙", elem_id="rf-theme-toggle", scale=0, min_width=42)

        dark_btn.click(
            fn=None, inputs=None, outputs=None,
            js="""() => {
                document.querySelector('body').classList.toggle('dark');
                const btn = document.querySelector('#rf-theme-toggle');
                btn.textContent = document.querySelector('body').classList.contains('dark') ? '☀️' : '🌙';
            }"""
        )

        with gr.Accordion("Settings", open=False):
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

        with gr.Row(elem_id="rf-toolbar"):
            clear_btn = gr.Button("🗑 Clear Chat", elem_id="rf-clear-btn", elem_classes=["rf-tool-btn"], scale=1)
            flush_btn = gr.Button("Flush KV Cache", elem_id="rf-flush-btn", elem_classes=["rf-tool-btn"], scale=1)

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

        def toggle_active(text):
            is_valid = bool(text and text.strip())
            return gr.update(elem_classes=["rf-action-btn", "rf-active" if is_valid else "rf-inactive"])

        msg.change(toggle_active, [msg], [send_btn], queue=False)

        def user_submit(message, chat_history):
            if not message or not message.strip():
                raise gr.Error("Please enter a message.")
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": message})
            return "", chat_history

        def bot_respond(chat_history, expert_name, model_name, endpoint_url):
            if not chat_history:
                return
            user_msg = chat_history[-1]["content"]
            prev = chat_history[:-1]
            try:
                for updated in respond(user_msg, prev, expert_name, model_name, endpoint_url):
                    yield updated
            except Exception as e:
                chat_history.append({"role": "assistant", "content": f"**Fatal Error:** {e}"})
                yield chat_history

        def flush_kv_cache(model_name, endpoint_url):
            """Unload the model from Ollama to clear its KV cache."""
            import requests
            base = endpoint_url.rsplit("/api/", 1)[0] if "/api/" in endpoint_url else endpoint_url.rstrip("/")
            try:
                resp = requests.post(
                    f"{base}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=10
                )
                resp.raise_for_status()
                gr.Info(f"KV cache cleared — {model_name} unloaded.")
            except Exception as e:
                gr.Warning(f"Failed to flush KV cache: {e}")
            return []

        msg.submit(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_respond, [chatbot, expert_dd, model_tb, url_tb], chatbot
        )

        send_btn.click(
            user_submit, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_respond, [chatbot, expert_dd, model_tb, url_tb], chatbot
        )

        clear_btn.click(lambda: [], None, chatbot, queue=False)

        flush_btn.click(
            flush_kv_cache, [model_tb, url_tb], [chatbot], queue=False
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
