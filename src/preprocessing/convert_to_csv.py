import pandas as pd

df = pd.read_csv("data/raw/SMSSpamCollection", sep="\t", names=["label", "message"])

df.to_csv("data/processed/spam.csv", index=False)

print("Converted to spam.csv with shape:", df.shape)
