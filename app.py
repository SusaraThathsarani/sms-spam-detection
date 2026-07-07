import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

MAX_LEN = 100
MODELS = {
    "LSTM": "models/lstm_model.h5",
    "Simple RNN": "models/simple_rnn_model.h5",
}
TOKENIZER_PATH = "data/processed/tokenizer.json"


@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_word_index(tokenizer_path: str):
    tokenizer_data = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    return {token: int(idx) for token, idx in tokenizer_data.items()}


def preprocess(message: str, word_index: dict) -> np.ndarray:
    tokens = message.lower().split()
    sequence = [word_index.get(token, 1) for token in tokens]
    if len(sequence) >= MAX_LEN:
        sequence = sequence[:MAX_LEN]
    else:
        sequence = sequence + [0] * (MAX_LEN - len(sequence))
    return np.asarray([sequence], dtype=np.int32)


st.set_page_config(page_title="SMS Spam Detector", page_icon=None, layout="centered")

st.title("SMS Spam Detector")
st.caption("Type an SMS message below and classify it as Spam or Ham using a trained model.")

model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Pipeline**\n\n"
    "Raw text -> Clean text -> Tokenize & Pad -> Model -> Spam/Ham"
)

message = st.text_area("SMS message", height=120, placeholder="e.g. You won a free prize, claim now!")

if st.button("Predict", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        model = load_model(MODELS[model_name])
        word_index = load_word_index(TOKENIZER_PATH)
        X = preprocess(message, word_index)
        probability = float(model.predict(X, verbose=0).ravel()[0])
        is_spam = probability > 0.5

        if is_spam:
            st.error(f"SPAM  (confidence: {probability * 100:.1f}%)")
        else:
            st.success(f"HAM  (confidence: {(1 - probability) * 100:.1f}%)")

        st.progress(probability if is_spam else 1 - probability)

with st.expander("Try example messages"):
    examples = [
        "You won a free prize, claim now!",
        "Hi, how are you today?",
        "URGENT! Your account has been suspended. Click here to verify.",
        "Ok lar... Joking wif u oni...",
    ]
    for example in examples:
        st.code(example, language=None)
