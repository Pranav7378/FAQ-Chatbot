import gradio as gr
from typing import List, Tuple
from rag_engine import ask_rag

# ---------------- Chat Function ----------------
def chat_with_rag(user_input: str, chat_history: List[Tuple[str, str]] = None):
    if chat_history is None:
        chat_history = []

    # Get answer from RAG
    answer = ask_rag(user_input)

    # Append to history
    chat_history.append((user_input, answer))

    # Clear input box
    return chat_history, ""

# ---------------- Gradio Interface ----------------
with gr.Blocks(theme=gr.themes.Soft(), title="Pranav Sai Portfolio Bot") as demo:
    gr.Markdown(
        """
        <div style="text-align:center; padding:15px; background-color:#f0f4f8; border-radius:12px; margin-bottom:20px;">
            <h1 style="color:#3b82f6; font-size:2.5em; font-weight:700;">Pranav Sai — Portfolio FAQ Bot</h1>
            <p style="color:#4b5563;">Ask about Pranav's skills, education, experience, or projects.</p>
        </div>
        """
    )

    chatbot = gr.Chatbot(
        label="Chat History",
        height=500,
        avatar_images=[None, "https://avatars.githubusercontent.com/u/108990666?v=4"],
        show_copy_button=True
    )

    with gr.Row():
        user_input = gr.Textbox(label="Ask a question", placeholder="e.g., What are Pranav's skills?", scale=4)
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.ClearButton([user_input, chatbot], value="Clear Chat")

    # Connect buttons
    submit_btn.click(chat_with_rag, inputs=[user_input, chatbot], outputs=[chatbot, user_input])
    user_input.submit(chat_with_rag, inputs=[user_input, chatbot], outputs=[chatbot, user_input])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
