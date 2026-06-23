import numpy as np
import pandas as pd

padded = np.load("data/processed/padded.npy")
labels = pd.read_csv("data/processed/labels.csv")

print("Shape:", padded.shape)
print("First row:", padded[0])
print("Labels shape:", labels.shape)
print(labels.head())
