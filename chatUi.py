import streamlit as st
from huggingface_hub import InferenceClient
import json
from analytics import Analytics

# Initialize Hugging Face client
client = InferenceClient(api_key="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")

def format_messages(conversation_history, current_prompt):
    """Format conversation history into proper alternating user/assistant format"""
    formatted_messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Please provide informative and accurate responses."
        }
    ]

    last_role = "system"
    for msg in conversation_history:
        if msg["role"] == "human" and last_role != "user":
            formatted_messages.append({
                "role": "user",
                "content": msg["content"]
            })
            last_role = "user"
        elif msg["role"] == "AI" and last_role != "system":
            formatted_messages.append({
                "role": "system",
                "content": msg["content"]
            })
            last_role = "system"

    if last_role != "user":
        formatted_messages.append({
            "role": "user",
            "content": current_prompt
        })

    return formatted_messages

def generate_text(prompt, conversation_history=[], analytics=None):
    """Generate text using Mixtral model with conversation history"""
    messages = format_messages(conversation_history, prompt)

    try:
        stream = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=True
        )
        
        result = ''
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, 'content'):
                result += chunk.choices[0].delta.content
        analytics.log_chat_usage(st.session_state.user_id, len(result.split()))
        return result
    
    except Exception as e:
        st.error(f"Error in generate_text: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def render_chat_header():
    """Render the chat interface header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1rem;'>
                <h1 style='color: #1E88E5; font-size: 2.5rem;'>💬 Messenger Prime</h1>
               
            </div>
        """, unsafe_allow_html=True)
    with col3:
        if st.button("🔄 New Chat", key="clear_chat_button", help="Start a new conversation"):
            st.session_state.messages = [{"role": "AI", "content": "👋 Hello! I'm Messenger Prime. How can I assist you today?"}]
            st.rerun()

def ChatUI(db):
    """Main chat interface"""
    analytics = Analytics(db)
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stChat {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #E3F2FD;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            max-width: 80%;
            float: right;
        }
        .assistant-message {
            background-color: #61676b;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            max-width: 80%;
            float: left;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .chat-container {
            height: 70vh;
            overflow-y: auto;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .chat-input {
            background-color: white;
            border-radius: 25px;
            padding: 10px 20px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    render_chat_header()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "AI", "content": "👋 Hello! I'm Messenger Prime. How can I assist you today?"}
        ]

    # Chat container
    with st.container():
        for msg in st.session_state.messages:
            message_class = "user-message" if msg["role"] == "human" else "assistant-message"
            st.markdown(f"""
                <div class="{message_class}">
                    {msg["content"]}
                </div>
                <div style='clear: both;'></div>
            """, unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("Type your message here...", key="chat_input")
    if prompt:
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.markdown(f"""
            <div class="user-message">
                {prompt}
            </div>
            <div style='clear: both;'></div>
        """, unsafe_allow_html=True)

        conversation_history = st.session_state.messages[-10:]
        
        with st.spinner("🤔 Thinking..."):
            try:
                response = generate_text(prompt, conversation_history, analytics)
                st.session_state.messages.append({"role": "AI", "content": response})
                st.markdown(f"""
                    <div class="assistant-message">
                        {response}
                    </div>
                    <div style='clear: both;'></div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Messenger Prime",
        page_icon="💭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    ChatUI()