import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to processed + labelled CSV")
parser.add_argument("--num_detectors", type=int, default=300, help="Number of candidate detectors to generate")
parser.add_argument("--radius", type=float, default=0.15, help="Detector radius for matching")
parser.add_argument("--output", required=True, help="Path to save results JSON")
args = parser.parse_args()

np.random.seed(42)

df = pd.read_csv(args.input)
features = ["cpu", "mem", "net"]

X = df[features].values
y_true = df["label"].values

# self set: normal (baseline) data points
self_set = X[y_true == 0]


def matches_self(detector, radius, self_set):
    """Check if a detector matches any point in the self set."""
    distances = np.linalg.norm(self_set - detector, axis=1)
    return np.any(distances < radius)


def generate_detectors(self_set, num_candidates, radius, n_features):
    """Generate detectors via negative selection (reject those matching self)."""
    detectors = []
    attempts = 0
    max_attempts = num_candidates * 20  # avoid infinite loop

    while len(detectors) < num_candidates and attempts < max_attempts:
        candidate = np.random.uniform(0, 1, n_features)
        if not matches_self(candidate, radius, self_set):
            detectors.append(candidate)
        attempts += 1

    return np.array(detectors)


print(f"Generating detectors (candidates={args.num_detectors}, radius={args.radius})...")
detectors = generate_detectors(self_set, args.num_detectors, args.radius, len(features))
print(f"Matured {len(detectors)} detectors from {args.num_detectors} candidates")


def detect_anomalies(X, detectors, radius):
    """Classify each sample: anomaly if any detector matches it."""
    predictions = np.zeros(len(X))
    for i, sample in enumerate(X):
        distances = np.linalg.norm(detectors - sample, axis=1)
        if np.any(distances < radius):
            predictions[i] = 1
    return predictions


print("Running detection...")
y_pred = detect_anomalies(X, detectors, args.radius).astype(int)

# evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

results = {
    "method": "nsa",
    "num_detectors_generated": len(detectors),
    "num_detectors_requested": args.num_detectors,
    "radius": args.radius,
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

print(f"\n=== NSA Results ===")
print(f"Detectors: {len(detectors)} (radius={args.radius})")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"Results saved to {args.output}")
