# 🤖 AI Safety POC — Chatbot with Abuse & Crisis Detection

This project is a Proof-of-Concept (POC) AI chatbot designed to demonstrate real-time **abuse**, **crisis**, and **escalation** detection using NLP models. It uses:

- 🧠 **OpenAI GPT (chat completions)**  
- 🤗 **Hugging Face Transformers for sentiment, toxicity, and hate speech detection**  
- 🎛️ **Streamlit** for interactive UI

> ⚠️ This project is intended for research and prototyping only. Heuristic-based crisis detection is **not suitable for production use without further validation**.

---

## 🔍 Features

- 🔐 Age-based **content filtering**
- 🚨 Heuristic **crisis detection** (e.g., self-harm or suicidal ideation)
- 🗯️ **Toxicity/abuse detection** using pre-trained models
- 📈 **Escalation monitoring** across conversation history
- 💬 Chat interface with **labels**: `SAFE`, `CRISIS`, `ABUSE`, `ESCALATION`
- 🧪 Debug mode to inspect underlying model predictions
- 📤 Export chat history in JSON format

---

## 📸 Demo

<img src="https://user-images.githubusercontent.com/your-screenshot-url" width="600" alt="AI Safety Chatbot Demo" />

---

## 🛠️ Setup

### 1. Clone the repository

2. Install dependencies

    pip install -r requirements.txt

Add your OpenAI API key
export OPENAI_API_KEY=your-api-key

Run the App

streamlit run safety_pipeline.py
Model Details
Task	Model

Sentiment	distilbert-base-uncased-finetuned-sst-2-english
Abuse/Toxicity	unitary/toxic-bert (fallbacks to sentiment model if loading fails)
Hate Speech Filter	facebook/roberta-hate-speech-dynabench-r4-target (optional)
Chat Completion	gpt-3.5-turbo via OpenAI API

.
├── safety_pipeline.py       # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md                # Project description



