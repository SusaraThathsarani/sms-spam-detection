import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

df = pd.read_csv("data/processed/cleaned.csv")

df["label_num"] = df["label"].map({"ham":0, "spam":1})

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_message"])
sequences = tokenizer.texts_to_sequences(df["clean_message"])

padded = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")

np.save("data/processed/padded.npy", padded)
df[["label_num"]].to_csv("data/processed/labels.csv", index=False)

print("Tokenized + padded dataset saved")
