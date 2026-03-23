import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
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
parser.add_argument("--model", choices=["linear", "ridge"], default="ridge", help="Regression model type")
parser.add_argument("--ridge-alpha", type=float, default=1.0, help="L2 regularization strength for Ridge")
parser.add_argument("--stress-weight", type=float, default=2.0, help="Sample weight multiplier for cpu phase targets")
parser.add_argument("--mixed-weight", type=float, default=1.5, help="Sample weight multiplier for mixed phase targets")
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


def fit_linear_calibrator(y_true: np.ndarray, y_pred: np.ndarray):
    """Fit y_true ~= a*y_pred + b per feature for simple calibration."""
    n_features = y_true.shape[1]
    alpha = np.ones(n_features, dtype=float)
    beta = np.zeros(n_features, dtype=float)

    for j in range(n_features):
        x = y_pred[:, j]
        y = y_true[:, j]
        if len(x) < 5 or np.std(x) < 1e-8:
            continue
        a, b = np.polyfit(x, y, 1)
        alpha[j] = float(a)
        beta[j] = float(b)

    return {"alpha": alpha.tolist(), "beta": beta.tolist()}


def apply_calibration(y_pred: np.ndarray, calibration: dict) -> np.ndarray:
    alpha = np.array(calibration.get("alpha", [1.0] * y_pred.shape[1]), dtype=float)
    beta = np.array(calibration.get("beta", [0.0] * y_pred.shape[1]), dtype=float)
    return (y_pred * alpha) + beta


def train_node_model(node_df, window, horizon):
    """Train a linear regression model for a single node."""
    values = node_df[FEATURES].values

    if len(values) < (window + horizon + 10):
        print(f"  Warning: only {len(values)} raw rows, skipping")
        return None, None, None

    # time split on raw sequence first; scaler must only see training period
    train_end_row = int(len(values) * 0.8)
    train_end_row = max(train_end_row, window + horizon)
    train_end_row = min(train_end_row, len(values) - 1)

    # fit scaler on TRAIN rows only to prevent leakage from test period
    scaler = MinMaxScaler()
    scaler.fit(values[:train_end_row])
    scaled = scaler.transform(values)

    # create sequences
    X, y = create_sequences(scaled, window, horizon)
    target_row_idx = np.array([i + window + horizon - 1 for i in range(len(X))])

    phase_col = None
    for c in ["phase_name", "phase", "phase_type"]:
        if c in node_df.columns:
            phase_col = c
            break

    target_phase = np.array(["unknown"] * len(X), dtype=object)
    if phase_col is not None:
        target_phase = node_df.iloc[target_row_idx][phase_col].astype(str).str.lower().values

    if len(X) < 10:
        print(f"  Warning: only {len(X)} samples, skipping")
        return None, None, None

    # split by whether target timestamp is in training period
    train_mask = target_row_idx < train_end_row
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    phase_train = target_phase[train_mask]

    if len(X_train) < 10 or len(X_test) < 1:
        print(f"  Warning: insufficient split after leakage-safe partition (train={len(X_train)}, test={len(X_test)}), skipping")
        return None, None, None

    # reserve a tail of the train period for calibration
    calib_start = max(10, int(len(X_train) * 0.8))
    X_fit, y_fit = X_train[:calib_start], y_train[:calib_start]
    X_cal, y_cal = X_train[calib_start:], y_train[calib_start:]
    phase_fit = phase_train[:calib_start]

    sample_weight = np.ones(len(X_fit), dtype=float)
    sample_weight[np.isin(phase_fit, ["cpu"]) ] = float(args.stress_weight)
    sample_weight[np.isin(phase_fit, ["mixed"]) ] = float(args.mixed_weight)

    # train model
    if args.model == "ridge":
        model = Ridge(alpha=float(args.ridge_alpha))
    else:
        model = LinearRegression()

    start = time.time()
    model.fit(X_fit, y_fit, sample_weight=sample_weight)
    train_time = time.time() - start

    if len(X_cal) >= 10:
        cal_pred = model.predict(X_cal)
        calibration = fit_linear_calibrator(y_cal, cal_pred)
    else:
        calibration = {"alpha": [1.0] * len(FEATURES), "beta": [0.0] * len(FEATURES)}

    # evaluate on test set
    start = time.time()
    y_pred = model.predict(X_test)
    y_pred = apply_calibration(y_pred, calibration)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    inference_time = (time.time() - start) / len(X_test) * 1000  # ms per prediction

    # metrics (on scaled data)
    mae = np.mean(np.abs(y_test - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))

    metrics = {
        "samples_total": len(X),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "samples_fit": len(X_fit),
        "samples_calibration": len(X_cal),
        "train_time_s": round(train_time, 4),
        "inference_time_ms": round(inference_time, 4),
        "mae_per_feature": {f: round(v, 4) for f, v in zip(FEATURES, mae)},
        "rmse_per_feature": {f: round(v, 4) for f, v in zip(FEATURES, rmse)},
        "mae_mean": round(float(mae.mean()), 4),
        "rmse_mean": round(float(rmse.mean()), 4)
    }

    bundle = {
        "model": model,
        "model_type": args.model,
        "calibration": calibration,
        "window": window,
        "horizon": horizon,
        "features": FEATURES,
    }

    return bundle, scaler, metrics


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

    model_bundle, scaler, metrics = train_node_model(node_df, args.window, args.horizon)

    if model_bundle is None:
        continue

    # save model and scaler
    node_short = node.replace("k3s-", "")
    model_path = os.path.join(args.output_dir, f"model_{node_short}.pkl")
    scaler_path = os.path.join(args.output_dir, f"scaler_{node_short}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
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
    "model": args.model,
    "ridge_alpha": args.ridge_alpha,
    "stress_weight": args.stress_weight,
    "mixed_weight": args.mixed_weight,
    "window": args.window,
    "horizon": args.horizon,
    "features": FEATURES,
    "nodes": all_metrics
}
summary_path = os.path.join(args.output_dir, "training_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Training complete. Summary saved to {summary_path}")
