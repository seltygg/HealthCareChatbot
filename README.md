# 🩺 HealthCare Chatbot

A machine‑learning–driven chatbot that provides **symptom‑based triage advice**.  
The core model is built with **scikit‑learn** decision trees, augmented by **AI agents** and **Retrieval-Augmented Generation (RAG)** via LangChain, with a Streamlit UI demo for zero‑boilerplate deployment.

---

## ✨ Features

- **Interactive dialogue** that adapts to user input, asks follow-up questions, and suggests possible conditions.  
- **Hybrid ML + AI pipeline**: Decision Trees for triage, LangChain agents for dynamic medical lookups.  
- **Pluggable front‑end**: CLI or Streamlit one‑page app with RAG, no changes to core code.  
- **Environment ready**: Dockerfile and `requirements.txt` for reproducible setup.  
- **MIT-licensed**—fork, customize, and contribute!

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/seltygg/HealthCareChatbot.git
cd HealthCareChatbot
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### 2 . Set your OpenAI API key
```bash
OPENAI_API_KEY=your_api_key_here
Or export it manually:
export OPENAI_API_KEY=your_api_key_here
```
### 3 . Run the CLI demo

```bash
python chat_bot.py
```
### 4. Launch the Streamlit UI
```bash
streamlit run app_streamlit.py
```