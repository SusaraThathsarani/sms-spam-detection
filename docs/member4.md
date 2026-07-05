# Member 4: Integration & Documentation

## What this part adds
- A single entry-point pipeline runner for preprocessing, training, and evaluation.
- A lightweight inference script to load the trained model and classify new messages.
- Updated project documentation for setup, execution, and deployment-style usage.

## How to use it
```bash
python src/integration/run_pipeline.py
python src/integration/predict.py --messages "You won a free prize" "Hi, how are you today?"
```
