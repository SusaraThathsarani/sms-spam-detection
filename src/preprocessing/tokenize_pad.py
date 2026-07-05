import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
except ModuleNotFoundError:  # pragma: no cover - environment fallback
    pad_sequences = None
    Tokenizer = None


class SimpleTokenizer:
    def __init__(self, num_words=5000, oov_token="<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}
        self.index_word = {1: oov_token}

    def fit_on_texts(self, texts):
        vocab = {}
        for text in texts:
            for token in str(text).split():
                vocab[token] = vocab.get(token, 0) + 1

        for token in sorted(vocab):
            if len(self.word_index) >= self.num_words:
                break
            self.word_index[token] = len(self.word_index) + 1
            self.index_word[self.word_index[token]] = token

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = str(text).split()
            sequence = [self.word_index.get(token, 1) for token in tokens]
            sequences.append(sequence)
        return sequences


def _pad_sequences_fallback(sequences, maxlen, padding="post", truncating="post"):
    padded = []
    for seq in sequences:
        if len(seq) >= maxlen:
            seq = seq[:maxlen]
        else:
            if padding == "post":
                seq = seq + [0] * (maxlen - len(seq))
            else:
                seq = [0] * (maxlen - len(seq)) + seq
        padded.append(seq)
    return np.asarray(padded, dtype=np.int32)


def prepare_sequences(texts, labels, max_len=100, vocab_size=5000):
    if Tokenizer is not None and pad_sequences is not None:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    else:
        tokenizer = SimpleTokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = _pad_sequences_fallback(sequences, max_len, padding="post", truncating="post")

    label_values = np.asarray(labels, dtype=np.int32)
    return padded, label_values, tokenizer


def build_processed_dataset(input_path="data/processed/cleaned.csv", output_path="data/processed/processed.csv"):
    df = pd.read_csv(input_path)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

    padded, labels, tokenizer = prepare_sequences(df["clean_message"].tolist(), df["label_num"].tolist())

    df["padded_sequence"] = [json.dumps(seq.tolist()) for seq in padded]
    df[["label", "message", "clean_message", "label_num", "padded_sequence"]].to_csv(output_path, index=False)

    np.save("data/processed/padded.npy", padded)
    pd.DataFrame({"label_num": labels}).to_csv("data/processed/labels.csv", index=False)

    tokenizer_path = Path("data/processed/tokenizer.json")
    tokenizer_path.write_text(json.dumps(tokenizer.word_index), encoding="utf-8")
    print(f"Processed dataset saved to {output_path}")
    return output_path


def main() -> None:
    build_processed_dataset()


if __name__ == "__main__":
    main()
