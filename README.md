# 🩺 HealthCare Chatbot

A machine‑learning–driven chatbot that provides **symptom‑based triage advice**.  
The core model is built with **scikit‑learn**, while optional UI layers let you deploy the bot as either:

- a **Streamlit one‑page web app** (zero‑boilerplate demo)

---

## ✨ Features
- **Interactive dialogue** that asks follow‑up questions and suggests possible conditions.
- **Lightweight ML pipeline** (TF‑IDF + Logistic Regression) trained on the included datasets.
- **Pluggable front‑end**: swap CLI ↔︎ Streamlit with no change to model code.
- **Docker‑ready** configuration for reproducible local or cloud deployment.
- **MIT‑licensed**—fork away!

---


---

## 🚀 Quick Start

### 1 . Clone & install
```bash
git clone https://github.com/seltygg/HealthCareChatbot.git
cd HealthCareChatbot
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```
### 2 . Run in the terminal (CLI)
```bash
python chat_bot.py
```
### 3 . Launch the Streamlit UI
```bash
python app_streamlit.py          # or: streamlit run app_streamlit.py
Then open the printed local URL in your browser.
```
