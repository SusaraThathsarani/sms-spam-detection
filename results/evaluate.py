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
    import os
    import json
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


    def plot_confusion_matrix(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Ham (0)', 'Spam (1)'])
        plt.yticks(tick_marks, ['Ham (0)', 'Spam (1)'])

        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved confusion matrix: {filename}")


    def plot_history(history_data, model_name, out_png):
        # history_data should be a dict with keys: accuracy, val_accuracy, loss, val_loss
        epochs = range(1, len(history_data.get('accuracy', [])) + 1)
        if len(epochs) == 0:
            print(f"No history to plot for {model_name}")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, history_data.get('accuracy', []), 'bo-', label='Train Acc')
        ax1.plot(epochs, history_data.get('val_accuracy', []), 'ro-', label='Val Acc')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(epochs, history_data.get('loss', []), 'bo-', label='Train Loss')
        ax2.plot(epochs, history_data.get('val_loss', []), 'ro-', label='Val Loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.suptitle(f"{model_name} Training History")
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved training history plot: {out_png}")


    def load_processed_data():
        # Prefer CSV processed file if exists, else fallback to padded numpy + labels
        csv_path = 'data/processed.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # If the CSV contains precomputed features, it's project-specific; prefer padded.npy when available
            # fall through to check numpy files

        # Fallback to padded numpy and labels.csv
        padded_path = 'data/processed/padded.npy'
        labels_path = 'data/processed/labels.csv'
        if os.path.exists(padded_path) and os.path.exists(labels_path):
            X = np.load(padded_path)
            labels_df = pd.read_csv(labels_path)
            if 'label_num' in labels_df.columns:
                y = labels_df['label_num'].values
            elif 'label' in labels_df.columns:
                y = labels_df['label'].map({'ham': 0, 'spam': 1}).fillna(labels_df['label']).astype(int).values
            else:
                # assume single column
                y = labels_df.iloc[:, 0].values
            return X, y

        raise FileNotFoundError('Processed data not found. Expected data/processed/padded.npy and data/processed/labels.csv')


    def evaluate_model(model_path, X_test, y_test):
        print(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        # Predict
        y_prob = model.predict(X_test, verbose=0)

        # Handle different output shapes
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            # multiclass probabilities -> take argmax
            y_pred = np.argmax(y_prob, axis=1)
        else:
            y_pred = (y_prob.ravel() > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        return y_pred, {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}


    def find_models():
        # Prefer official spam_model.h5
        preferred = 'models/spam_model.h5'
        if os.path.exists(preferred):
            return [preferred]

        # Otherwise search for any keras/h5 files in models/
        files = []
        for ext in ('*.h5', '*.keras'):
            files.extend(glob.glob(os.path.join('models', ext)))
        return files


    def main():
        print('Loading processed data...')
        X, y = load_processed_data()
        # Use 20% as test set with fixed seed to match training
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f'Test set size: {X_test.shape[0]}')

        models = find_models()
        if not models:
            print('No models found in models/. Place trained model as models/spam_model.h5 or .keras/.h5 files.')
            return

        os.makedirs('results', exist_ok=True)
        summary = {}

        for mp in models:
            name = os.path.splitext(os.path.basename(mp))[0]
            try:
                y_pred, metrics = evaluate_model(mp, X_test, y_test)
                summary[name] = metrics

                cm_file = f'results/{name}_confusion_matrix.png'
                plot_confusion_matrix(y_test, y_pred, f'{name} Confusion Matrix', cm_file)

                # Try to find a history JSON in results/ or models/
                history_paths = [f'results/{name}_history.json', f'models/{name}_history.json', f'results/{name}.history.json']
                history = None
                for hp in history_paths:
                    if os.path.exists(hp):
                        with open(hp, 'r') as f:
                            history = json.load(f)
                        break
                if history:
                    hist_png = f'results/{name}_training.png'
                    plot_history(history, name, hist_png)
                else:
                    print(f'No training history found for {name}, skipping history plots.')

            except Exception as e:
                print(f'Error evaluating {mp}: {e}')

        # Save metrics summary
        with open('results/metrics_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        print('Saved results/metrics_summary.json')

        # Write a brief report
        with open('results/report.md', 'w') as f:
            f.write('# Evaluation Report\n\n')
            if not summary:
                f.write('No model evaluations were completed.\n')
            else:
                f.write('## Summary Metrics\n\n')
                f.write('| Model | Accuracy | Precision | Recall | F1-Score |\n')
                f.write('|---|---:|---:|---:|---:|\n')
                for m, met in summary.items():
                    f.write(f'| {m} | {met["Accuracy"]:.4f} | {met["Precision"]:.4f} | {met["Recall"]:.4f} | {met["F1-Score"]:.4f} |\n')
                f.write('\n')
                f.write('Plots saved in the `results/` folder.\n')

        print('Wrote results/report.md')


    if __name__ == '__main__':
        main()
