import numpy as np
import pandas as pd

df = pd.read_csv("data/processed.csv")
X = df[["cpu", "mem", "net"]].values

# normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

train = X[:1000]
test = X[1000:]

detectors = []
radius = 0.5
num_detectors = 50

def matches(det, x):
    return np.linalg.norm(det - x) < radius

np.random.seed(0)
while len(detectors) < num_detectors:
    cand = np.random.uniform(-3, 3, size=3)
    if not any(matches(cand, v) for v in train):
        detectors.append(cand)

labels = []
for x in test:
    labels.append(int(any(matches(d, x) for d in detectors)))

df["nsa_anomaly"] = [0]*1000 + labels
df.to_csv("data/nsa_output.csv", index=False)
print("NSA detection complete. Results saved to data/nsa_output.csv")