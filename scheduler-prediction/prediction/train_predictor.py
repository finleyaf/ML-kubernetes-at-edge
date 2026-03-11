import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import argparse
import os
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to raw dataset CSV (from netdata collector)")
parser.add_argument("--window", type=int, default=10, help="Sliding window size (number of past timesteps)")
parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon (steps ahead)")
parser.add_argument("--output-dir", required=True, help="Directory to save trained model and scaler")
args = parser.parse_args()

FEATURES = ["cpu_user", "cpu_system", "ram_used", "net_received", "net_sent"]


def create_sequences(data, window, horizon):
    """Create sliding window input/output pairs for training.

    For each window of `window` timesteps, predict the values
    `horizon` steps into the future.
    """
    X, y = [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i:i + window].flatten())
        y.append(data[i + window + horizon - 1])
    return np.array(X), np.array(y)


def train_node_model(node_df, window, horizon):
    """Train a linear regression model for a single node."""
    values = node_df[FEATURES].values

    # fit scaler on this node's data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # create sequences
    X, y = create_sequences(scaled, window, horizon)

    if len(X) < 10:
        print(f"  Warning: only {len(X)} samples, skipping")
        return None, None, None

    # train/test split (80/20, no shuffle for time series)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # train model
    model = LinearRegression()
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # evaluate on test set
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start) / len(X_test) * 1000  # ms per prediction

    # metrics (on scaled data)
    mae = np.mean(np.abs(y_test - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))

    metrics = {
        "samples_total": len(X),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "train_time_s": round(train_time, 4),
        "inference_time_ms": round(inference_time, 4),
        "mae_per_feature": {f: round(v, 4) for f, v in zip(FEATURES, mae)},
        "rmse_per_feature": {f: round(v, 4) for f, v in zip(FEATURES, rmse)},
        "mae_mean": round(float(mae.mean()), 4),
        "rmse_mean": round(float(rmse.mean()), 4)
    }

    return model, scaler, metrics


# load data
df = pd.read_csv(args.input)
print(f"Loaded {len(df)} rows from {args.input}")
print(f"Window: {args.window}, Horizon: {args.horizon}")
print(f"Features: {FEATURES}")
print()

os.makedirs(args.output_dir, exist_ok=True)
all_metrics = {}

for node in df["node"].unique():
    # skip control node
    if "control" in node:
        continue

    print(f"Training model for {node}...")
    node_df = df[df["node"] == node].sort_values("timestamp").reset_index(drop=True)

    # check required columns exist
    missing = [f for f in FEATURES if f not in node_df.columns]
    if missing:
        print(f"  Missing columns: {missing}, skipping")
        continue

    model, scaler, metrics = train_node_model(node_df, args.window, args.horizon)

    if model is None:
        continue

    # save model and scaler
    node_short = node.replace("k3s-", "")
    model_path = os.path.join(args.output_dir, f"model_{node_short}.pkl")
    scaler_path = os.path.join(args.output_dir, f"scaler_{node_short}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    all_metrics[node] = metrics

    print(f"  MAE:  {metrics['mae_mean']:.4f}")
    print(f"  RMSE: {metrics['rmse_mean']:.4f}")
    print(f"  Inference: {metrics['inference_time_ms']:.4f} ms/prediction")
    print(f"  Saved: {model_path}")
    print()

# save training summary
summary = {
    "window": args.window,
    "horizon": args.horizon,
    "features": FEATURES,
    "nodes": all_metrics
}
summary_path = os.path.join(args.output_dir, "training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Training complete. Summary saved to {summary_path}")
