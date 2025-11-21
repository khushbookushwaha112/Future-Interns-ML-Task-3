import streamlit as st
import pandas as pd
from openai import OpenAI

# ------------------------------
# SET YOUR OPENAI API KEY HERE
# ------------------------------
client = OpenAI(api_key="YOUR_API_KEY_HERE")

# ------------------------------
# STREAMLIT APP TITLE
# ------------------------------
st.title("💬 Customer Support Chatbot")
st.write("This AI-powered chatbot answers customer support questions using a Kaggle dataset.")

# ------------------------------
# LOAD DATASET
# ------------------------------
try:
    data = pd.read_csv("dataset/customer_support_cleaned.csv")
except:
    st.error("Dataset not found! Make sure 'dataset/customer_support_cleaned.csv' exists.")
    st.stop()

# ------------------------------
# CONVERT DATA INTO FAQ TEXT
# ------------------------------
faq_text = ""
for index, row in data.iterrows():
    faq_text += f"Q: {row['question']}\nA: {row['answer']}\n\n"

# ------------------------------
# FUNCTION TO GENERATE RESPONSE
# ------------------------------
def get_response(user_query):
    prompt = f"""
You are a customer support assistant.
Use the FAQ dataset below to respond:

FAQ Knowledge Base:
{faq_text}

User question: {user_query}

If the answer is not available in the dataset,
try to give the best helpful customer-support reply.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful support assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"❌ Error: {e}"

# ------------------------------
# INPUT BOX
# ------------------------------
user_input = st.text_input("Ask a customer support question:")

# ------------------------------
# SHOW CHATBOT RESPONSE
# ------------------------------
if user_input:
    answer = get_response(user_input)
    st.write("### 🤖 Chatbot Response:")
    st.write(answer)
