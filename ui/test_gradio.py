import gradio as gr
import time

def gen(t):
    yield "Start"
    time.sleep(2)
    yield "Done"

def post_gen(text):
    is_valid = bool(text and text.strip())
    print("post_gen called with:", repr(text), "is_valid:", is_valid)
    return [
        gr.update(visible=True, interactive=is_valid),
        gr.update(visible=False)
    ]

def pre_gen():
    return [
        gr.update(visible=False),
        gr.update(visible=True, interactive=True)
    ]

with gr.Blocks() as demo:
    with gr.Row():
        msg = gr.Textbox()
        send_btn = gr.Button("Play")
        stop_btn = gr.Button("Stop", visible=False)
    
    out = gr.Textbox()

    submit_1 = send_btn.click(pre_gen, None, [send_btn, stop_btn], queue=False)
    submit_2 = submit_1.then(gen, msg, out)
    submit_2.then(post_gen, msg, [send_btn, stop_btn], queue=False)

    stop_btn.click(post_gen, msg, [send_btn, stop_btn], queue=False, cancels=[submit_2])

demo.launch(server_port=7000, prevent_thread_lock=True)
import time
time.sleep(10)
