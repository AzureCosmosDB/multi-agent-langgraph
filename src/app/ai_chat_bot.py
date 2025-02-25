import gradio as gr
import uuid
from langchain.schema import AIMessage
from multi_agent_service import graph

# Set local_interactive_mode to False in multi_agent_service
import multi_agent_service

multi_agent_service.local_interactive_mode = False

# Global variable to persist session ID
current_session_id = None


def chatbot(input_text, session_id, messages):
    """Process user input and return AI response in OpenAI-style format."""
    global current_session_id
    if not session_id or not isinstance(session_id, str):
        session_id = current_session_id if current_session_id else str(uuid.uuid4())
    current_session_id = session_id  # Store the session ID for future use

    # Set local_interactive_mode to True before streaming the graph
    multi_agent_service.local_interactive_mode = True

    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
    input_message = {"messages": [{"role": "user", "content": input_text}]}
    response = {"role": "assistant", "content": ""}

    messages.append({"role": "user", "content": input_text})

    for update in graph.stream(input_message, config=config, stream_mode="updates"):
        for _, value in update.items():
            if isinstance(value, dict) and value.get("messages"):
                last_message = value["messages"][-1]
                if isinstance(last_message, AIMessage):
                    response["content"] = last_message.content

    messages.append({"role": "assistant", "content": response["content"]})
    return messages


# Define Gradio UI
with gr.Blocks(css=".chatbox { background-color: #f9f9f9; border-radius: 10px; padding: 10px; }") as demo:
    gr.Markdown(
        """
        # Personal Shopping AI Assistant
        Welcome to your Personal Shopping AI Assistant. 
        Get help with shopping, refunds, product information, and more!
        """,
        elem_id="header",
    )

    chatbot_display = gr.Chatbot(label="Chat with the Assistant", elem_classes=["chatbox"], type="messages")

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Enter your message here...",
            label="Your Message",
            lines=1,
            elem_id="user_input",
        )

    session_id = gr.State(str(uuid.uuid4()))
    messages = gr.State([])

    # Chat interaction
    user_input.submit(
        fn=chatbot,
        inputs=[user_input, session_id, messages],
        outputs=[chatbot_display],
    ).then(
        lambda: "", inputs=None, outputs=user_input
    )  # Clear the input box after submission

demo.launch()
