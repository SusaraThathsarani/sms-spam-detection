import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_script(script_rel_path: str) -> None:
    script_path = REPO_ROOT / script_rel_path
    subprocess.run([sys.executable, str(script_path)], cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SMS spam detection pipeline end to end.")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip dataset preprocessing steps")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation and report generation")
    args = parser.parse_args()

    if not args.skip_preprocessing:
        run_script("src/preprocessing/convert_to_csv.py")
        run_script("src/preprocessing/clean_text.py")
        run_script("src/preprocessing/tokenize_pad.py")

    if not args.skip_training:
        try:
            import tensorflow  # noqa: F401
        except ModuleNotFoundError:
            print("TensorFlow is not installed. Install the dependencies from requirements.txt to train the model.")
            return 1
        run_script("src/models/train.py")

    if not args.skip_evaluation:
        if not (REPO_ROOT / "models" / "spam_model.h5").exists():
            print("No trained model was found. Run the pipeline with training enabled or place a model at models/spam_model.h5.")
            return 1
        run_script("results/evaluate.py")

    print("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
