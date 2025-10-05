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

**2. Install dependencies**

    pip install -r requirements.txt

Add your OpenAI API key in .env file
export OPENAI_API_KEY=your-api-key

Run the App

streamlit run safety_pipeline.py


How It Works

**🔍 1. Sentiment, Abuse & Hate Speech Detection**

Loads Hugging Face pipelines for:

Sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english)

Abuse detection (unitary/toxic-bert)

Hate speech detection (facebook/roberta-hate-speech-dynabench-r4-target)

Fallback logic: if a model fails to load, the system defaults to a simpler one.

**🚨 2. Crisis Detection**

Uses heuristic keyword matching (e.g., "suicide", "want to die") and strong negative sentiment to detect potential mental health crises.

If detected, the system prompts the AI to respond with empathy and encourage help-seeking behavior.

**📈 3. Escalation Detection**

Analyzes the last few user messages to identify increasing negativity or toxic language.

If escalation is detected, the AI responds calmly to de-escalate the conversation.

**🔞 4. Age-Based Content Filtering

Blocks content labeled as hate speech if the user’s age is below 18.

Ensures safer interactions for younger users.

**🤖 5. Response Generation with GPT-3.5**

Uses OpenAI’s GPT-3.5-turbo to generate assistant replies.

Includes system prompts tailored to the context (e.g., safe, abusive, crisis, escalating).

**🏷️ 6. Response Labeling**

Each response is labeled as one of:

SAFE — Normal conversation

ABUSE — Toxic or abusive language detected

CRISIS — Potential mental health crisis

ESCALATION — Increasing negativity/toxicity

