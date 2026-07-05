# Intelligent SMS Spam Detection using RNN and LSTM

## 📌 Objectives
- Build an NLP pipeline for SMS spam detection.
- Use RNN and LSTM to capture sequential dependencies in text.
- Achieve >95% accuracy on the UCI SMS Spam dataset.

## 📊 Dataset
- UCI SMS Spam Collection (5,574 messages: 4,827 ham, 747 spam).

## 🛠️ Tech Stack
- Python 3.9+
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn, Matplotlib

## 📂 Repo Structure
- `data/` → datasets
- `src/` → source code
- `docs/` → documentation

## 🚀 Getting Started
```bash
git clone https://github.com/SusaraThathsarani/sms-spam-detection.git
cd sms-spam-detection
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Use the integration entry point to run preprocessing, training, and evaluation in one command:
```bash
python src/integration/run_pipeline.py
```

### 3. Run Inference on New Messages
```bash
python src/integration/predict.py --messages "You won a free prize" "Hi, how are you today?"
```

## 🔮 Deployment & Inference
The trained model is saved as [models/spam_model.h5](models/spam_model.h5). The inference script loads that model and classifies new SMS messages using the saved tokenizer vocabulary from [data/processed/tokenizer.json](data/processed/tokenizer.json).

## 📝 License
This project is open-source and available under the MIT License.
