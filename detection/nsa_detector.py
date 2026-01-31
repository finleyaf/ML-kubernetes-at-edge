import numpy as np
import pandas as pd

df = pd.read_csv("data/processed.csv")
X = df[["cpu", "mem", "net"]].values

# normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)

train = X[df["label"] == 0]
test = X[1000:]

detectors = []
radius = 1.5
num_detectors = 200

# distance of anomalies to nearest normal point
normal = X[df["label"] == 0]
anomalies = X[df["label"] == 1]

dists = []
for a in anomalies:
    dists.append(np.min(np.linalg.norm(normal - a, axis=1)))

print("Min distance anomaly→normal:", np.min(dists))
print("Mean distance anomaly→normal:", np.mean(dists))

def matches(det, x):
    return np.linalg.norm(det - x) < radius

np.random.seed(0)
while len(detectors) < num_detectors:
    cand = np.random.uniform(-3, 3, size=3)
    if not any(matches(cand, v) for v in train):
        detectors.append(cand)

print("Number of detectors generated:", len(detectors))

labels = []
for x in test:
    labels.append(int(any(matches(d, x) for d in detectors)))

df["nsa_anomaly"] = [0]*1000 + labels
print("Total NSA anomalies:", sum(df["nsa_anomaly"]))
df.to_csv("data/nsa_output.csv", index=False)
print("NSA detection complete. Results saved to data/nsa_output.csv")