import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set page config
st.set_page_config(page_title="ChatGPT (Dev)", page_icon="💬", layout="centered")

# Title and description
st.title("💬 ChatGPT - Local Dev Version")
st.markdown("A private ChatGPT built using Hugging Face's `DialoGPT`. Ideal for development use.")

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize chat history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []
if "generated_responses" not in st.session_state:
    st.session_state.generated_responses = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Encode user input + add eos token
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append tokens to history
    bot_input_ids = (
        torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
        if st.session_state.chat_history_ids is not None
        else new_input_ids
    )

    # Generate response
    output_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75
    )

    # Extract and decode response
    response = tokenizer.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update history
    st.session_state.chat_history_ids = output_ids
    st.session_state.past_inputs.append(user_input)
    st.session_state.generated_responses.append(response)

# Display chat history
for i in range(len(st.session_state.past_inputs)):
    st.markdown(f"**You:** {st.session_state.past_inputs[i]}")
    st.markdown(f"**Bot:** {st.session_state.generated_responses[i]}")

# Clear button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history_ids = None
    st.session_state.past_inputs = []
    st.session_state.generated_responses = []
