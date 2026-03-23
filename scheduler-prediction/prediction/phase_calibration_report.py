import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from predictor import ClusterPredictor, FEATURES


def resolve_phase_column(df: pd.DataFrame) -> str:
    for c in ["phase_name", "phase", "phase_type"]:
        if c in df.columns:
            return c
    return ""


def safe_polyfit(x: np.ndarray, y: np.ndarray):
    if len(x) < 8 or float(np.std(x)) < 1e-8:
        return 1.0, 0.0, 0.0
    a, b = np.polyfit(x, y, 1)
    y_hat = (a * x) + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - (ss_res / ss_tot))
    return float(a), float(b), float(r2)


def ece(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    err = 0.0
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        n = int(np.sum(mask))
        if n == 0:
            continue
        mean_p = float(np.mean(y_pred[mask]))
        mean_t = float(np.mean(y_true[mask]))
        err += (n / total) * abs(mean_t - mean_p)
    return float(err)


def feature_metrics(y_true: np.ndarray, y_pred: np.ndarray, bins: int) -> Dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    bias = float(np.mean(y_pred - y_true))
    slope, intercept, r2 = safe_polyfit(y_pred, y_true)
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "bias": round(bias, 6),
        "calibration_slope": round(slope, 6),
        "calibration_intercept": round(intercept, 6),
        "calibration_r2": round(r2, 6),
        "ece": round(ece(y_true, y_pred, bins=bins), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-wise calibration report for predictor models")
    parser.add_argument("--input", required=True, help="Path to labelled CSV")
    parser.add_argument("--model-dir", required=True, help="Model directory")
    parser.add_argument("--window", type=int, required=True, help="Window size used by predictor")
    parser.add_argument("--horizon", type=int, required=True, help="Prediction horizon")
    parser.add_argument("--bins", type=int, default=10, help="ECE bins")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    phase_col = resolve_phase_column(df)
    if not phase_col:
        raise RuntimeError("No phase column found. Expected one of: phase_name, phase, phase_type")

    cluster = ClusterPredictor(args.model_dir, window_size=args.window)
    for node in sorted(df["node"].unique()):
        if "control" in str(node).lower():
            continue
        try:
            cluster.add_node(str(node))
        except FileNotFoundError:
            continue

    report = {
        "window": int(args.window),
        "horizon": int(args.horizon),
        "features": FEATURES,
        "phase_column": phase_col,
        "nodes": {},
    }

    for node_name, predictor in cluster.predictors.items():
        node_df = df[df["node"] == node_name].sort_values("timestamp").reset_index(drop=True)

        rows: List[Dict] = []
        for i in range(len(node_df)):
            row = node_df.iloc[i]
            obs = {f: float(row[f]) for f in FEATURES}

            if predictor.ready() and i + args.horizon < len(node_df):
                pred = predictor.predict()
                if pred is not None:
                    fut = node_df.iloc[i + args.horizon]
                    y_true = predictor.scaler.transform(np.array([[float(fut[f]) for f in FEATURES]]))[0]
                    phase = str(fut[phase_col]).strip().lower()
                    rows.append(
                        {
                            "phase": phase if phase else "unknown",
                            "y_true": y_true.tolist(),
                            "y_pred": [float(pred[f]) for f in FEATURES],
                        }
                    )

            predictor.update(obs)

        if not rows:
            continue

        phases = sorted(set(r["phase"] for r in rows))
        node_out = {"num_predictions": len(rows), "phases": {}}

        # overall metrics
        y_true_all = np.array([r["y_true"] for r in rows], dtype=float)
        y_pred_all = np.array([r["y_pred"] for r in rows], dtype=float)
        node_out["overall"] = {
            f: feature_metrics(y_true_all[:, j], y_pred_all[:, j], bins=args.bins)
            for j, f in enumerate(FEATURES)
        }

        for ph in phases:
            ph_rows = [r for r in rows if r["phase"] == ph]
            y_true = np.array([r["y_true"] for r in ph_rows], dtype=float)
            y_pred = np.array([r["y_pred"] for r in ph_rows], dtype=float)
            node_out["phases"][ph] = {
                "count": len(ph_rows),
                "features": {
                    f: feature_metrics(y_true[:, j], y_pred[:, j], bins=args.bins)
                    for j, f in enumerate(FEATURES)
                },
            }

        report["nodes"][node_name] = node_out

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"saved {args.output}")
    print(f"nodes: {list(report['nodes'].keys())}")


if __name__ == "__main__":
    main()
