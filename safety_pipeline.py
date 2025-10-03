import os
import streamlit as st
from typing import List, Tuple, Dict, Any
from transformers import pipeline
import openai
import json

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="🤖 AI Safety POC", page_icon="🤖", layout="wide")

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Set in environment or Streamlit secrets.")
    st.stop()
openai.api_key = OPENAI_API_KEY

# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis",
                         model="distilbert-base-uncased-finetuned-sst-2-english",
                         truncation=True)
    try:
        abuse = pipeline("text-classification", model="unitary/toxic-bert", truncation=True)
    except Exception:
        abuse = sentiment
    try:
        content = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", truncation=True)
    except Exception:
        content = abuse
    return abuse, sentiment, content

abuse_model, sentiment_model, content_model = load_models()

# ----------------------------
# Heuristics
# ----------------------------
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "want to die", "end my life", "hang myself",
    "hurt myself", "i cant go on", "i'm done", "i am done", "self harm", "kill me"
]

def detect_crisis(text: str) -> Tuple[bool, Dict[str, Any]]:
    t = text.lower()
    for k in CRISIS_KEYWORDS:
        if k in t:
            return True, {"reason": "keyword", "keyword": k}
    s = sentiment_model(text)[0]
    a = abuse_model(text)[0]

    if any(k in a.get("label", "").lower() for k in ["toxic", "abusive"]) and a.get("score", 0) > 0.8:
        return False, {}  # strictly abuse, not crisis

    if s.get("label", "").lower().startswith("neg") and s.get("score", 0) >= 0.92:
        return True, {"reason": "very_negative_sentiment", "score": s.get("score")}

    return False, {}

def detect_escalation(history: List[Tuple[str, str]], window: int = 4) -> Tuple[bool, Dict[str, Any]]:
    recent_user_msgs = [u for u, b in history if u][-window:]
    if len(recent_user_msgs) < 2:
        return False, {"reason": "not_enough_msgs"}

    neg_count, tox_count = 0, 0
    neg_scores, tox_scores = [], []

    for m in recent_user_msgs:
        s = sentiment_model(m)[0]
        a = abuse_model(m)[0]
        neg_scores.append((s.get("label"), s.get("score")))
        tox_scores.append((a.get("label"), a.get("score")))

        if s.get("label", "").lower().startswith("neg") and s.get("score", 0) > 0.6:
            neg_count += 1
        if any(k in a.get("label", "").lower() for k in ["toxic", "abusive"]) and a.get("score", 0) > 0.6:
            tox_count += 1

    increasing_neg = False
    seq = [(-s if lab.lower().startswith("neg") else s) for lab, s in neg_scores]
    if len(seq) >= 2 and seq[-1] < seq[0] and seq[-1] < -0.5:
        increasing_neg = True

    if neg_count >= 2 or tox_count >= 2 or increasing_neg:
        return True, {"neg_count": neg_count, "tox_count": tox_count, "neg_scores": neg_scores, "tox_scores": tox_scores}

    return False, {"neg_count": neg_count, "tox_count": tox_count}

def build_history_str(history: List[Tuple[str, str]]) -> str:
    return "\n".join([f"User: {u}\nAssistant: {b}" for u, b in history if u])

# ----------------------------
# OpenAI LLM wrapper
# ----------------------------
def generate_llm_response(system_prompt: str, history_str: str, user_input: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_str:
        messages.append({"role": "system", "content": f"Conversation so far:\n{history_str}"})
    messages.append({"role": "user", "content": user_input})

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"

# ----------------------------
# Chatbot orchestration
# ----------------------------
def chatbot_response(user_msg: str, user_age: int, history: List[Tuple[str, str]]) -> Dict[str, Any]:
    abuse = abuse_model(user_msg)[0]
    sentiment = sentiment_model(user_msg)[0]
    content = content_model(user_msg)[0]

    # Age-based content filter
    if ("hate" in content.get("label", "").lower() or "hate_speech" in content.get("label", "").lower()) and user_age < 18:
        return {"type": "CONTENT_FILTER", "message": "⚠️ This content is blocked for your age.", "meta": {"content": content}}

    # Crisis detection
    is_crisis, crisis_meta = detect_crisis(user_msg)
    if is_crisis:
        system_prompt = "You are calm and empathetic. Respond sensitively, encourage seeking help."
        response_type = "CRISIS"
    # Abuse detection
    elif any(k in abuse.get("label", "").lower() for k in ["toxic", "abusive"]) and abuse.get("score", 0) > 0.8:
        system_prompt = "You are firm but polite. The user used abusive language. De-escalate and set boundaries."
        response_type = "ABUSE"
    else:
        system_prompt = "You are a helpful, polite assistant. Keep responses concise and friendly."
        response_type = "SAFE"

    # Escalation detection
    hist_for_check = history + [(user_msg, None)]
    is_escalating, esc_meta = detect_escalation(hist_for_check)
    if is_escalating and response_type not in ["CRISIS", "ABUSE"]:
        response_type = "ESCALATION"
        system_prompt = "You are calm. Respond with de-escalation steps, acknowledge feelings, and stay neutral."

    history_str = build_history_str(history)
    bot_msg = generate_llm_response(system_prompt, history_str, user_msg)

    meta = {
        "abuse": abuse,
        "sentiment": sentiment,
        "content": content,
        "crisis_meta": crisis_meta,
        "escalation_meta": esc_meta
    }

    return {"type": response_type, "message": bot_msg, "meta": meta}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🤖 AI Safety POC — Chat Simulator with Labels")
st.write("Abuse, crisis, and escalation detection enabled. General queries get SAFE label.")

# User settings
user_age = st.sidebar.number_input("Your age", min_value=10, max_value=100, value=18)
show_meta = st.sidebar.checkbox("Show debug meta", value=False)

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
user_input = st.chat_input("Type a message...")
if user_input:
    st.session_state.history.append((user_input, None))

# Render chat with labels in bold
for i, (u_msg, b_msg) in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(u_msg)

    if b_msg is None:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("...thinking")
            bot_resp = chatbot_response(
                u_msg,
                user_age,
                [(um, bm.get("message") if isinstance(bm, dict) else bm) for um, bm in st.session_state.history if um]
            )
            # Show label in bold above message
            label_text = f"**Label: {bot_resp.get('type', 'SAFE')}**"
            placeholder.markdown(f"{label_text}\n\n{bot_resp['message']}")
            st.session_state.history[i] = (u_msg, bot_resp)
    else:
        with st.chat_message("assistant"):
            label_text = f"**Label: {b_msg.get('type', 'SAFE')}**" if isinstance(b_msg, dict) else ""
            st.markdown(f"{label_text}\n\n{b_msg.get('message') if isinstance(b_msg, dict) else str(b_msg)}")

    if show_meta and isinstance(b_msg, dict):
        with st.expander("Debug meta"):
            st.json(b_msg.get("meta", {}))

# Sidebar controls
st.sidebar.markdown("---")
if st.sidebar.button("Clear chat"):
    st.session_state.history = []
    st.experimental_rerun()

if st.sidebar.button("Export chat (JSON)"):
    js = json.dumps([
        {"user": u, "assistant": (b.get("message") if isinstance(b, dict) else b),
         "meta": (b.get("meta") if isinstance(b, dict) else {})}
        for u, b in st.session_state.history
    ], indent=2)
    st.download_button("Download conversation JSON", js, file_name="conversation.json", mime="application/json")

st.markdown("---")
st.markdown("**Note:** Crisis detection is heuristic. Replace with trained models for production use.")
