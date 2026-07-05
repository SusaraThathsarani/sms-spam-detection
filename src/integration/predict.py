import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:  # pragma: no cover - environment fallback
    raise SystemExit("TensorFlow is required for inference. Install requirements.txt first.") from exc


def load_model(model_path: str = "models/spam_model.h5"):
    return tf.keras.models.load_model(model_path)


def tokenize_messages(messages, tokenizer_path: str = "data/processed/tokenizer.json"):
    tokenizer_data = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    word_index = {token: int(idx) for token, idx in tokenizer_data.items()}

    sequences = []
    for message in messages:
        tokens = str(message).split()
        sequence = [word_index.get(token, 1) for token in tokens]
        sequences.append(sequence)

    max_len = 100
    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [0] * (max_len - len(seq)))
    return np.asarray(padded, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with the trained SMS spam detector.")
    parser.add_argument("--messages", nargs="+", required=True, help="One or more SMS messages to classify")
    parser.add_argument("--model", default="models/spam_model.h5", help="Path to the trained model")
    args = parser.parse_args()

    model = load_model(args.model)
    X = tokenize_messages(args.messages)
    probabilities = model.predict(X, verbose=0).ravel()
    predictions = ["Spam" if prob > 0.5 else "Ham" for prob in probabilities]

    for message, prediction, probability in zip(args.messages, predictions, probabilities):
        print(f"{prediction}\t{probability:.4f}\t{message}")


if __name__ == "__main__":
    main()
