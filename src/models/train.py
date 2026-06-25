import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from sklearn.model_selection import train_test_split

def main():
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    X = np.load("data/processed/padded.npy")
    y = pd.read_csv("data/processed/labels.csv")["label_num"].values

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test/Validation shape: {X_test.shape}")

    vocab_size = 5000
    embedding_dim = 64
    max_len = 100

    # ------------------ 1. Simple RNN Model ------------------
    print("\n" + "="*50)
    print("1. Training Simple RNN Model")
    print("="*50)
    
    rnn_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True),
        SimpleRNN(64),
        Dense(1, activation='sigmoid')
    ])
    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    rnn_model.summary()

    print("Training Simple RNN Model...")
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    # Save Simple RNN model and history
    print("Saving Simple RNN Model and History...")
    rnn_model.save("models/simple_rnn_model.keras")
    rnn_model.save("models/simple_rnn_model.h5")
    with open("results/simple_rnn_history.json", "w") as f:
        json.dump(rnn_history.history, f)

    # ------------------ 2. LSTM Model ------------------
    print("\n" + "="*50)
    print("2. Training LSTM Model")
    print("="*50)
    
    lstm_model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.summary()

    print("Training LSTM Model...")
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    # Save LSTM model and history
    print("Saving LSTM Model and History...")
    lstm_model.save("models/lstm_model.keras")
    lstm_model.save("models/lstm_model.h5")
    with open("results/lstm_history.json", "w") as f:
        json.dump(lstm_history.history, f)

    print("\nTraining completed successfully for both RNN and LSTM models!")

if __name__ == "__main__":
    main()
