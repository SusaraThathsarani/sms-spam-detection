import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    # Custom color mapping and style
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Ham (0)', 'Spam (1)'], fontsize=11)
    plt.yticks(tick_marks, ['Ham (0)', 'Spam (1)'], fontsize=11)
    
    # Annotate matrix cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=14, fontweight='bold')
                 
    plt.ylabel('True Class', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {filename}")

def plot_training_history(history_file, title, filename):
    if not os.path.exists(history_file):
        print(f"Warning: History file {history_file} not found.")
        return
        
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Accuracy Plot
    ax1.plot(epochs, history['accuracy'], 'o-', label='Train Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, history['val_accuracy'], 's-', label='Val Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=10)
    
    # Loss Plot
    ax2.plot(epochs, history['loss'], 'o-', label='Train Loss', color='#d62728', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 's-', label='Val Loss', color='#2ca02c', linewidth=2)
    ax2.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved training history plot: {filename}")

def evaluate_model(model_path, X_test, y_test):
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Run predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return y_pred, acc, prec, rec, f1

def main():
    # Load dataset
    print("Loading test data...")
    X = np.load("data/processed/padded.npy")
    y = pd.read_csv("data/processed/labels.csv")["label_num"].values

    # Train-test split (80/20) with seed 42 to match training split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Evaluation Test Set Size: {X_test.shape[0]}")

    results = {}

    # Evaluate Simple RNN
    rnn_model_path = "models/simple_rnn_model.keras"
    if os.path.exists(rnn_model_path):
        y_pred, acc, prec, rec, f1 = evaluate_model(rnn_model_path, X_test, y_test)
        results['Simple RNN'] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
        plot_confusion_matrix(y_test, y_pred, "Simple RNN Confusion Matrix", "results/simple_rnn_confusion_matrix.png")
        plot_training_history("results/simple_rnn_history.json", "Simple RNN Training History", "results/simple_rnn_training.png")
    else:
        print(f"Error: {rnn_model_path} not found.")

    # Evaluate LSTM
    lstm_model_path = "models/lstm_model.keras"
    if os.path.exists(lstm_model_path):
        y_pred, acc, prec, rec, f1 = evaluate_model(lstm_model_path, X_test, y_test)
        results['LSTM'] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
        plot_confusion_matrix(y_test, y_pred, "LSTM Confusion Matrix", "results/lstm_confusion_matrix.png")
        plot_training_history("results/lstm_history.json", "LSTM Training History", "results/lstm_training.png")
    else:
        print(f"Error: {lstm_model_path} not found.")

    # Print summary table
    print("\n" + "="*60)
    print("                    EVALUATION METRICS SUMMARY")
    print("="*60)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} | {metrics['Accuracy']:<10.4f} | {metrics['Precision']:<10.4f} | {metrics['Recall']:<10.4f} | {metrics['F1-Score']:<10.4f}")
    print("="*60)

    # Save results summary to a JSON file for report generation
    with open("results/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved metrics summary to results/metrics_summary.json")

if __name__ == "__main__":
    main()
