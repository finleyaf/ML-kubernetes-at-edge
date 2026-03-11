import pandas as pd
import numpy as np
import json
import argparse
import os
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predictor import ClusterPredictor, FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to raw dataset CSV (unseen test data)")
parser.add_argument("--model-dir", required=True, help="Directory containing trained models")
parser.add_argument("--window", type=int, default=10, help="Sliding window size (must match training)")
parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon (must match training)")
parser.add_argument("--output", required=True, help="Path to save evaluation results JSON")
args = parser.parse_args()

df = pd.read_csv(args.input)
print(f"Loaded {len(df)} rows from {args.input}")

# initialise cluster predictor
cluster = ClusterPredictor(args.model_dir, window_size=args.window)

# add worker nodes
for node in df["node"].unique():
    if "control" in node:
        continue
    try:
        cluster.add_node(node)
    except FileNotFoundError as e:
        print(f"Skipping {node}: {e}")

results = {}

for node_name, predictor in cluster.predictors.items():
    node_df = df[df["node"] == node_name].sort_values("timestamp").reset_index(drop=True)

    actuals = []
    predictions = []
    inference_times = []

    for i in range(len(node_df)):
        row = node_df.iloc[i]
        observation = {f: row[f] for f in FEATURES}

        # if buffer is full and we have a future value to compare against
        if predictor.ready() and i + args.horizon < len(node_df):
            start = time.time()
            pred = predictor.predict()
            elapsed = (time.time() - start) * 1000  # ms

            if pred is not None:
                future_row = node_df.iloc[i + args.horizon]
                future_scaled = predictor.scaler.transform(
                    np.array([[future_row[f] for f in FEATURES]])
                )[0]

                predictions.append([pred[f] for f in FEATURES])
                actuals.append(future_scaled.tolist())
                inference_times.append(elapsed)

        # feed observation into buffer
        predictor.update(observation)

    if not predictions:
        print(f"{node_name}: no predictions generated")
        continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # per-feature metrics
    mae = np.mean(np.abs(actuals - predictions), axis=0)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2, axis=0))
    # use sMAPE (symmetric MAPE) to avoid division-by-zero when actuals are near 0
    smape = np.mean(
        2 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions) + 1e-8),
        axis=0
    ) * 100

    results[node_name] = {
        "num_predictions": len(predictions),
        "avg_inference_time_ms": round(float(np.mean(inference_times)), 4),
        "max_inference_time_ms": round(float(np.max(inference_times)), 4),
        "mae_per_feature": {f: round(v, 4) for f, v in zip(FEATURES, mae)},
        "rmse_per_feature": {f: round(v, 4) for f, v in zip(FEATURES, rmse)},
        "smape_per_feature": {f: round(v, 2) for f, v in zip(FEATURES, smape)},
        "mae_mean": round(float(mae.mean()), 4),
        "rmse_mean": round(float(rmse.mean()), 4),
        "smape_mean": round(float(smape.mean()), 2)
    }

    print(f"\n{node_name}:")
    print(f"  Predictions:    {len(predictions)}")
    print(f"  MAE (mean):     {results[node_name]['mae_mean']:.4f}")
    print(f"  RMSE (mean):    {results[node_name]['rmse_mean']:.4f}")
    print(f"  sMAPE (mean):   {results[node_name]['smape_mean']:.2f}%")
    print(f"  Inference time: {results[node_name]['avg_inference_time_ms']:.4f} ms (avg)")
    print(f"                  {results[node_name]['max_inference_time_ms']:.4f} ms (max)")

# save results
output = {
    "window": args.window,
    "horizon": args.horizon,
    "features": FEATURES,
    "nodes": results
}

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nEvaluation saved to {args.output}")
