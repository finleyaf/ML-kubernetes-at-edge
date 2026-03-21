import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold


FEATURES = ["cpu", "mem", "net"]


@dataclass
class Config:
    method: str
    window: int
    num_detectors: int = 300
    radius: float = 0.15
    threshold_rule: str = "mean2std"

    def key(self) -> str:
        if self.method == "nsa":
            return f"nsa_w{self.window}_d{self.num_detectors}_r{self.radius}"
        return f"kmeans_w{self.window}_{self.threshold_rule}"


def load_runs(runs_dir: str) -> pd.DataFrame:
    frames = []
    run_dirs = sorted(
        d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))
    )

    for run_id in run_dirs:
        p = os.path.join(runs_dir, run_id, "labelled.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df["run_id"] = run_id
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No labelled run files found in {runs_dir}")

    data = pd.concat(frames, ignore_index=True)
    needed = ["timestamp", "node", "cpu_user", "cpu_system", "ram_used", "net_received", "net_sent", "label", "run_id"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    return data


def preprocess_for_window(df: pd.DataFrame, window: int) -> pd.DataFrame:
    work = df.copy()
    work["cpu"] = work["cpu_user"] + work["cpu_system"]
    work["mem"] = work["ram_used"]
    work["net"] = work["net_received"] + work["net_sent"]

    frames = []
    for (run_id, node), g in work.groupby(["run_id", "node"], sort=False):
        g = g.sort_values("timestamp").copy()

        for f in FEATURES:
            g[f] = g[f].rolling(window=window, min_periods=1).mean()

        # min-max normalise within each run+node stream
        for f in FEATURES:
            mn = g[f].min()
            mx = g[f].max()
            if mx - mn > 0:
                g[f] = (g[f] - mn) / (mx - mn)
            else:
                g[f] = 0.0

        frames.append(g[["run_id", "timestamp", "node", *FEATURES, "label"]])

    return pd.concat(frames, ignore_index=True)


def get_splits(run_ids: List[str], strategy: str, n_splits: int) -> List[Tuple[List[str], List[str]]]:
    run_ids = sorted(run_ids)
    if strategy == "loo":
        return [([r for r in run_ids if r != test], [test]) for test in run_ids]

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for group-kfold")
    if n_splits > len(run_ids):
        n_splits = len(run_ids)

    # GroupKFold requires sample rows; emulate by indexing run list itself.
    X = np.arange(len(run_ids)).reshape(-1, 1)
    y = np.zeros(len(run_ids))
    groups = np.array(run_ids)

    folds = []
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X, y, groups):
        train_runs = [run_ids[i] for i in train_idx]
        test_runs = [run_ids[i] for i in test_idx]
        folds.append((train_runs, test_runs))
    return folds


def compute_threshold(distances: np.ndarray, rule: str) -> float:
    if rule == "mean2std":
        return float(distances.mean() + 2 * distances.std())
    if rule == "mean3std":
        return float(distances.mean() + 3 * distances.std())
    if rule == "p95":
        return float(np.percentile(distances, 95))
    if rule == "p99":
        return float(np.percentile(distances, 99))
    raise ValueError(f"Unknown threshold rule: {rule}")


def fit_predict_kmeans(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold_rule: str) -> np.ndarray:
    X_train = train_df[FEATURES].values
    X_test = test_df[FEATURES].values

    model = KMeans(n_clusters=2, random_state=42, n_init=10)
    model.fit(X_train)

    train_normal = train_df[train_df["label"] == 0]
    if len(train_normal) == 0:
        raise RuntimeError("No normal samples in training set for KMeans threshold")

    d_train_normal = model.transform(train_normal[FEATURES].values).min(axis=1)
    threshold = compute_threshold(d_train_normal, threshold_rule)

    d_test = model.transform(X_test).min(axis=1)
    return (d_test > threshold).astype(int)


def mature_detectors(self_set: np.ndarray, num_detectors: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    detectors = []
    attempts = 0
    max_attempts = num_detectors * 40

    while len(detectors) < num_detectors and attempts < max_attempts:
        cand = rng.uniform(0, 1, size=self_set.shape[1])
        d = np.linalg.norm(self_set - cand, axis=1)
        if not np.any(d < radius):
            detectors.append(cand)
        attempts += 1

    if not detectors:
        return np.empty((0, self_set.shape[1]))
    return np.array(detectors)


def fit_predict_nsa(train_df: pd.DataFrame, test_df: pd.DataFrame, num_detectors: int, radius: float) -> np.ndarray:
    self_set = train_df[train_df["label"] == 0][FEATURES].values
    if len(self_set) == 0:
        raise RuntimeError("No normal samples in training set for NSA")

    rng = np.random.default_rng(42)
    detectors = mature_detectors(self_set, num_detectors=num_detectors, radius=radius, rng=rng)
    if len(detectors) == 0:
        # if no detectors matured, model predicts normal only
        return np.zeros(len(test_df), dtype=int)

    X_test = test_df[FEATURES].values
    y_pred = np.zeros(len(X_test), dtype=int)
    for i, sample in enumerate(X_test):
        d = np.linalg.norm(detectors - sample, axis=1)
        if np.any(d < radius):
            y_pred[i] = 1
    return y_pred


def run_config(data_w: pd.DataFrame, config: Config, splits: List[Tuple[List[str], List[str]]]) -> List[Dict]:
    rows = []
    for fold_i, (train_runs, test_runs) in enumerate(splits, start=1):
        train_df = data_w[data_w["run_id"].isin(train_runs)]
        test_df_all = data_w[data_w["run_id"].isin(test_runs)]

        if config.method == "kmeans":
            y_pred_all = fit_predict_kmeans(train_df, test_df_all, config.threshold_rule)
        else:
            y_pred_all = fit_predict_nsa(train_df, test_df_all, config.num_detectors, config.radius)

        test_df_all = test_df_all.copy()
        test_df_all["y_pred"] = y_pred_all

        for run_id, run_df in test_df_all.groupby("run_id"):
            y_true = run_df["label"].values
            y_pred = run_df["y_pred"].values

            rows.append(
                {
                    "config_key": config.key(),
                    "method": config.method,
                    "window": config.window,
                    "num_detectors": config.num_detectors if config.method == "nsa" else None,
                    "radius": config.radius if config.method == "nsa" else None,
                    "threshold_rule": config.threshold_rule if config.method == "kmeans" else None,
                    "fold": fold_i,
                    "run_id": run_id,
                    "samples": int(len(run_df)),
                    "anomalies": int((y_true == 1).sum()),
                    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                }
            )

    return rows


def ci95(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    margin = 1.96 * sem
    return mean, mean - margin, mean + margin


def summarise(per_run_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for key, g in per_run_df.groupby("config_key"):
        p_mean, p_lo, p_hi = ci95(g["precision"].tolist())
        r_mean, r_lo, r_hi = ci95(g["recall"].tolist())
        f_mean, f_lo, f_hi = ci95(g["f1"].tolist())

        row0 = g.iloc[0]
        summary_rows.append(
            {
                "config_key": key,
                "method": row0["method"],
                "window": int(row0["window"]),
                "num_detectors": row0["num_detectors"],
                "radius": row0["radius"],
                "threshold_rule": row0["threshold_rule"],
                "runs_evaluated": int(g["run_id"].nunique()),
                "precision_mean": round(p_mean, 4),
                "precision_ci_low": round(p_lo, 4),
                "precision_ci_high": round(p_hi, 4),
                "recall_mean": round(r_mean, 4),
                "recall_ci_low": round(r_lo, 4),
                "recall_ci_high": round(r_hi, 4),
                "f1_mean": round(f_mean, 4),
                "f1_ci_low": round(f_lo, 4),
                "f1_ci_high": round(f_hi, 4),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["f1_mean", "recall_mean", "precision_mean"], ascending=False)


def build_config_grid(args) -> List[Config]:
    configs = []

    for w, rule in itertools.product(args.windows, args.kmeans_threshold_rules):
        configs.append(Config(method="kmeans", window=w, threshold_rule=rule))

    for w, d, r in itertools.product(args.windows, args.nsa_detectors, args.nsa_radius):
        configs.append(Config(method="nsa", window=w, num_detectors=d, radius=r))

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, help="Path to dataset/runs")
    parser.add_argument("--strategy", choices=["loo", "group-kfold"], default="loo")
    parser.add_argument("--n-splits", type=int, default=5, help="Used only for group-kfold")
    parser.add_argument("--windows", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--nsa-detectors", nargs="+", type=int, default=[200, 300])
    parser.add_argument("--nsa-radius", nargs="+", type=float, default=[0.1, 0.15, 0.2])
    parser.add_argument(
        "--kmeans-threshold-rules",
        nargs="+",
        default=["mean2std", "mean3std", "p95", "p99"],
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_runs(args.runs_dir)
    run_ids = sorted(data["run_id"].unique())
    splits = get_splits(run_ids, strategy=args.strategy, n_splits=args.n_splits)

    print(f"Loaded {len(run_ids)} runs and {len(data)} rows")
    print(f"Validation strategy: {args.strategy}, folds={len(splits)}")

    configs = build_config_grid(args)
    print(f"Evaluating {len(configs)} detector configs")

    preprocessed_by_window = {}
    for w in sorted(set(args.windows)):
        preprocessed_by_window[w] = preprocess_for_window(data, w)

    per_run_records = []
    for i, config in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {config.key()}")
        d_w = preprocessed_by_window[config.window]
        recs = run_config(d_w, config, splits)
        per_run_records.extend(recs)

    per_run_df = pd.DataFrame(per_run_records)
    summary_df = summarise(per_run_df)

    best = summary_df.iloc[0].to_dict()
    locked = {
        "selected_at": pd.Timestamp.now("UTC").isoformat(),
        "strategy": args.strategy,
        "n_splits": args.n_splits if args.strategy == "group-kfold" else None,
        "config": {
            "method": best["method"],
            "window": int(best["window"]),
            "num_detectors": None if pd.isna(best["num_detectors"]) else int(best["num_detectors"]),
            "radius": None if pd.isna(best["radius"]) else float(best["radius"]),
            "kmeans_threshold_rule": None if pd.isna(best["threshold_rule"]) else best["threshold_rule"],
        },
        "performance": {
            "f1_mean": float(best["f1_mean"]),
            "f1_ci_low": float(best["f1_ci_low"]),
            "f1_ci_high": float(best["f1_ci_high"]),
            "precision_mean": float(best["precision_mean"]),
            "recall_mean": float(best["recall_mean"]),
            "runs_evaluated": int(best["runs_evaluated"]),
        },
    }

    per_run_csv = os.path.join(args.output_dir, "per_run_metrics.csv")
    summary_csv = os.path.join(args.output_dir, "config_summary.csv")
    summary_json = os.path.join(args.output_dir, "validation_summary.json")
    locked_json = os.path.join(args.output_dir, "locked_detector_config.json")

    per_run_df.to_csv(per_run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump(
            {
                "strategy": args.strategy,
                "n_splits": args.n_splits if args.strategy == "group-kfold" else None,
                "runs_total": len(run_ids),
                "rows_total": int(len(data)),
                "configs_evaluated": len(configs),
                "best_config_key": best["config_key"],
                "best_method": best["method"],
            },
            f,
            indent=2,
        )

    with open(locked_json, "w") as f:
        json.dump(locked, f, indent=2)

    print("\n=== Detection Validation Complete ===")
    print(f"Per-run metrics:      {per_run_csv}")
    print(f"Config summary:       {summary_csv}")
    print(f"Validation summary:   {summary_json}")
    print(f"Locked detector conf: {locked_json}")
    print(f"Selected config: {best['config_key']} (f1_mean={best['f1_mean']})")


if __name__ == "__main__":
    main()
