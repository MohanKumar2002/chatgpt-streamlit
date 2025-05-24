import streamlit as st
from transformers import pipeline, set_seed

# Set Streamlit page config — must be first!
st.set_page_config(page_title="Dev ChatGPT", layout="wide")

# Load CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize model
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for chat history
with st.sidebar:
    st.title("💬 Chat History")
    if st.session_state.history:
        for i, (user, bot) in enumerate(st.session_state.history[::-1]):
            st.markdown(f"**You:** {user}")
            st.markdown(f"**GPT:** {bot}")
            st.markdown("---")
    else:
        st.info("No chats yet.")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

# Main UI
st.title("🧠 Dev ChatGPT")
st.subheader("Chat interface for building and testing models")

user_input = st.text_input("Ask something...", key="input")

if st.button("Send") and user_input:
    with st.spinner("Generating..."):
        response = generator(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
        st.session_state.history.append((user_input, response))

# Display last interaction
if st.session_state.history:
    user_msg, bot_msg = st.session_state.history[-1]
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**GPT:** {bot_msg}")
