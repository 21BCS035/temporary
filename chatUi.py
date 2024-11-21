import streamlit as st
from huggingface_hub import InferenceClient
import json
from analytics import Analytics

# Initialize Hugging Face client
client = InferenceClient(api_key="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")

def format_messages(conversation_history, current_prompt):
    """
    Format conversation history into proper alternating user/assistant format
    """
    formatted_messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Please provide informative and accurate responses."
        }
    ]

    # Process conversation history ensuring strict alternation
    last_role = "system"  # Start tracking from the system message
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

    # Add the current prompt as the final user message
    if last_role != "user":
        formatted_messages.append({
            "role": "user",
            "content": current_prompt
        })

    return formatted_messages

def generate_text(prompt, conversation_history=[], analytics=None):
    """
    Generate text using Mixtral model with conversation history
    """
    # Format messages in the correct alternating pattern
    messages = format_messages(conversation_history, prompt)

    try:
        # Generate response
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
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("💬 Mixtral Chatbot")
        st.caption("🚀 A Streamlit chatbot powered by Mixtral-8x7B-Instruct")
    with col2:
        if st.button("Clear Chat", key="clear_chat_button"):
            st.session_state.messages = [{"role": "AI", "content": "👋 Hello! How can I help you today?"}]
            st.rerun()

def ChatUI(db):
    """Main chat interface"""
    analytics = Analytics(db)
    render_chat_header()

    # Initialize messages in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "AI", "content": "👋 Hello! How can I help you today?"}
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input and response generation
    if prompt := st.chat_input():
        # Add user message to chat
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("user").write(prompt)

        # Get conversation history (last 10 messages for context)
        conversation_history = st.session_state.messages[-10:]
        
        # Generate response with spinner
        with st.spinner("🧠 Thinking..."):
            try:
                response = generate_text(prompt, conversation_history, analytics)
                
                # Add AI response to chat
                st.session_state.messages.append({"role": "AI", "content": response})
                st.chat_message("assistant").write(response)
            
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Mixtral Chatbot",
        page_icon="💭",
        layout="wide"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .stChat {
            padding: 20px;
        }
        .stChatMessage {
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    ChatUI()