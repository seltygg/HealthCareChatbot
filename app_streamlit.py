import streamlit as st
from chat_bot import get_response

st.title("AI‑Powered Healthcare Chatbot")

prompt = st.text_area(
    "Describe your symptoms & questions",
    placeholder="e.g. 'fever, headache; days=3. Any precautions?'"
)

if st.button("Ask CareBot"):
    if not prompt.strip():
        st.error("Please provide some input.")
    else:
        answer = get_response(prompt)
        st.markdown(answer)
