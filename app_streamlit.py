import streamlit as st
from chat_bot import get_chatbot_response, chatbot  # our new module

st.title("Healthcare Chatbot")

# 1) Primary symptom
initial = st.text_input("Primary symptom (e.g. fever):")

# 2) Duration
days = st.number_input(
    "How many days have you been experiencing this?", 
    min_value=1, max_value=30, value=1
)

# 3) Other symptoms
additional = st.multiselect(
    "Any other symptoms?", 
    options=chatbot.symptoms, 
    help="Start typing to filter."
)

# 4) Trigger prediction
if st.button("Get Diagnosis"):
    if not initial:
        st.error("Please enter at least one symptom.")
    else:
        resp = get_chatbot_response(initial, days, additional)
        st.subheader("👩‍⚕️ You may have:")
        st.write(resp["disease"])
        st.subheader("⚠️ Risk Assessment:")
        st.write(resp["risk_advice"])
        st.subheader("📝 Description:")
        st.write(resp["description"])
        st.subheader("💡 Precautions:")
        for idx, p in enumerate(resp["precautions"], 1):
            st.write(f"{idx}. {p}")
