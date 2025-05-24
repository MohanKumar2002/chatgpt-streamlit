import streamlit as st
from transformers import pipeline, set_seed
import time
import os

# Apply custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App Configuration
st.set_page_config(page_title="Dev ChatGPT", layout="wide")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "microsoft/DialoGPT-medium"

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.model_name = st.selectbox(
        "Choose a model", 
        ["microsoft/DialoGPT-medium", "gpt2", "distilgpt2"], 
        index=0
    )
    
    st.markdown("---")
    st.markdown("## 🕘 Chat History")
    for i, (user, bot) in enumerate(st.session_state.history):
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Bot:** {bot}")
    if st.button("🧹 Clear History"):
        st.session_state.history = []

# Header
st.markdown("<h1 class='title'>🤖 Dev ChatGPT - Model Testing Playground</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Test, fine-tune, and build your own chat models</p>", unsafe_allow_html=True)

# Load model
generator = pipeline("text-generation", model=st.session_state.model_name)
set_seed(42)

# Chat Input
user_input = st.chat_input("Type your message...")

if user_input:
    with st.spinner("Thinking..."):
        response = generator(user_input, max_length=150, num_return_sequences=1, do_sample=True)
        full_response = response[0]['generated_text']
        cleaned = full_response[len(user_input):].strip()

        st.chat_message("user").write(user_input)
        st.chat_message("assistant").write(cleaned)

        st.session_state.history.append((user_input, cleaned))
