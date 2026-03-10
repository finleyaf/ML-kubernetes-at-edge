import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to processed + labelled CSV")
parser.add_argument("--k", type=int, default=2, help="Number of clusters")
parser.add_argument("--output", required=True, help="Path to save results JSON")
args = parser.parse_args()

df = pd.read_csv(args.input)
features = ["cpu", "mem", "net"]

X = df[features].values
y_true = df["label"].values

# train k-means
kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
kmeans.fit(X)

# compute distances to nearest cluster centre
distances = kmeans.transform(X).min(axis=1)

# set threshold as mean + 2 * std of distances from baseline (normal) points
baseline_mask = y_true == 0
threshold = distances[baseline_mask].mean() + 2 * distances[baseline_mask].std()

# classify: points beyond threshold are anomalies
y_pred = (distances > threshold).astype(int)

# evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

results = {
    "method": "k-means",
    "k": args.k,
    "threshold": float(threshold),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "true_positives": int(tp),
    "false_positives": int(fp),
    "true_negatives": int(tn),
    "false_negatives": int(fn),
    "total_samples": len(y_true),
    "total_anomalies": int(y_true.sum()),
    "detected_anomalies": int(y_pred.sum())
}

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n=== K-Means Results (k={args.k}) ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Threshold: {threshold:.4f}")
print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"Results saved to {args.output}")
