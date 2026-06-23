import pandas as pd
import re

df = pd.read_csv("data/processed/spam.csv")

def clean_message(msg):
    msg = msg.lower()
    msg = re.sub(r"[^a-z0-9\s]", "", msg)  
    return msg

df["clean_message"] = df["message"].apply(clean_message)
df.to_csv("data/processed/cleaned.csv", index=False)

print("Cleaned dataset saved to cleaned.csv")
