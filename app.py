import streamlit as st
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

st.title("💬 Customer Support Chatbot (Future Interns Task 3)")
st.write("Ask your customer support questions!")

data = pd.read_csv("dataset/customer_support_cleaned.csv")

faq_text = ""
for index, row in data.iterrows():
    faq_text += f"Q: {row['question']} \nA: {row['answer']}\n\n"

def get_response(user_query):
    prompt = f"""
You are a customer support assistant.
Use the FAQ dataset to help the user.

FAQ Knowledge Base:
{faq_text}

User question: {user_query}

If answer is not in FAQs, politely reply based on your customer-care knowledge.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful support assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message["content"]

user_input = st.text_input("Ask something:")

if user_input:
    answer = get_response(user_input)
    st.write("### 🤖 Response:")
    st.write(answer)
