import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

print(df.head())
print(df.shape)
