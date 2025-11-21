import streamlit as st
import pandas as pd
from transformers import pipeline

# ----------------------------
# Load HuggingFace Model
# ----------------------------
chat_model = pipeline("text-generation", model="distilgpt2")

# ----------------------------
# Streamlit Title
# ----------------------------
st.title("💬 Customer Support Chatbot (AI-Powered)")
st.write("This chatbot uses an AI model + FAQ dataset to answer customer support questions.")

# ----------------------------
# Load Dataset
# ----------------------------
try:
    data = pd.read_csv("dataset/customer_support_cleaned.csv")
except:
    st.error("❌ Dataset missing! Please add 'dataset/customer_support_cleaned.csv'")
    st.stop()

# ----------------------------
# Convert Dataset to FAQ Text
# ----------------------------
faq_text = ""
for _, row in data.iterrows():
    faq_text += f"Q: {row['question']}\nA: {row['answer']}\n\n"

# ----------------------------
# Chatbot Logic
# ----------------------------
def ai_answer(user_query):
    prompt = (
        "You are a polite and helpful customer support assistant.\n\n"
        "Here is the company's FAQ knowledge base:\n"
        f"{faq_text}\n\n"
        f"User Question: {user_query}\n"
        "Answer the user based on the FAQs or general customer support knowledge.\n"
    )

    ai_output = chat_model(prompt, max_length=200, num_return_sequences=1)
    return ai_output[0]["generated_text"]

# ----------------------------
# Text Input
# ----------------------------
user_input = st.text_input("Ask your customer support question:")

# ----------------------------
# Display Answer
# ----------------------------
if user_input:
    response = ai_answer(user_input)
    st.subheader("🤖 Chatbot Response:")
    st.write(response)
