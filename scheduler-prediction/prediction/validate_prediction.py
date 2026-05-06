import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


FEATURES = ["cpu_user", "cpu_system", "ram_used", "net_received", "net_sent"]
MODELS = ["linear", "naive_last", "naive_mean"]


def parse_run_number(run_id: str) -> int:
    try:
        return int(run_id.split("_")[-1])
    except Exception:
        return 0


def assign_phase_name(ts: float, phases: List[Dict]) -> str:
    for idx, phase in enumerate(phases):
        start = float(phase["start"])
        end = float(phase["end"])
        is_last = idx == len(phases) - 1
        in_range = (start <= ts <= end) if is_last else (start <= ts < end)
        if in_range:
            phase_type = str(phase.get("type", "")).strip().lower()
            phase_name = str(phase.get("name", phase_type)).strip().lower()
            if phase_type == "stress" and phase_name:
                return phase_name
            if phase_type:
                return phase_type
            if phase_name:
                return phase_name
            return "unknown"
    return "unknown"


def load_runs(runs_dir: str) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], List[str], Dict[str, set]]:
    run_dirs = sorted(
        d for d in os.listdir(runs_dir) if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))
    )
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {runs_dir}")

    data_by_run_node: Dict[str, Dict[str, pd.DataFrame]] = {}
    run_phase_sets: Dict[str, set] = {}

    for run_id in run_dirs:
        labelled_path = os.path.join(runs_dir, run_id, "labelled.csv")
        phases_path = os.path.join(runs_dir, run_id, "phases.json")

        if not os.path.exists(labelled_path) or not os.path.exists(phases_path):
            continue

        df = pd.read_csv(labelled_path)
        with open(phases_path, "r") as f:
            phases = json.load(f)

        needed = ["timestamp", "node", *FEATURES]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise RuntimeError(f"{run_id}: missing required columns: {missing}")

        df = df.copy()
        df["phase_name"] = df["timestamp"].apply(lambda ts: assign_phase_name(ts, phases))
        df["run_id"] = run_id
        run_phase_sets[run_id] = set(df["phase_name"].dropna().astype(str).tolist())

        node_map: Dict[str, pd.DataFrame] = {}
        for node, node_df in df.groupby("node"):
            node_name = str(node)
            if "control" in node_name.lower():
                continue
            node_map[node_name] = node_df.sort_values("timestamp").reset_index(drop=True)

        if node_map:
            data_by_run_node[run_id] = node_map

    ordered_runs = sorted(data_by_run_node.keys(), key=parse_run_number)
    if not ordered_runs:
        raise RuntimeError(f"No valid runs with data found in {runs_dir}")

    return data_by_run_node, ordered_runs, run_phase_sets


def select_stratified_holdout(ordered_runs: List[str], run_phase_sets: Dict[str, set], holdout_count: int) -> List[str]:
    if holdout_count >= len(ordered_runs):
        raise RuntimeError(f"holdout_count={holdout_count} must be smaller than total runs={len(ordered_runs)}")

    stress_runs = [r for r in ordered_runs if {"cpu", "mixed"}.intersection(run_phase_sets.get(r, set()))]
    control_runs = [r for r in ordered_runs if "recovery" in run_phase_sets.get(r, set()) and r not in stress_runs]

    target_stress = min(len(stress_runs), max(1, holdout_count // 2))
    target_control = min(len(control_runs), holdout_count - target_stress)

    selected = []
    selected.extend(stress_runs[-target_stress:])
    selected.extend(control_runs[-target_control:])

    if len(selected) < holdout_count:
        remaining = [r for r in ordered_runs if r not in selected]
        selected.extend(remaining[-(holdout_count - len(selected)):])

    selected = sorted(set(selected), key=parse_run_number)
    if len(selected) > holdout_count:
        selected = selected[-holdout_count:]
    return selected


def create_sequences(data: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, idx = [], [], []
    for i in range(len(data) - window - horizon + 1):
        X.append(data[i:i + window].flatten())
        target_idx = i + window + horizon - 1
        y.append(data[target_idx])
        idx.append(target_idx)
    return np.array(X), np.array(y), np.array(idx)


def train_linear_model(train_frames: List[pd.DataFrame], window: int, horizon: int) -> Tuple[Optional[LinearRegression], Optional[MinMaxScaler]]:
    if not train_frames:
        return None, None

    values = np.concatenate([f[FEATURES].values for f in train_frames], axis=0)
    if len(values) < (window + horizon + 10):
        return None, None

    scaler = MinMaxScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    X_train, y_train, _ = create_sequences(scaled, window, horizon)
    if len(X_train) < 10:
        return None, None

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, scaler


def predict_with_model(
    model_name: str,
    node_df: pd.DataFrame,
    window: int,
    horizon: int,
    scaler: MinMaxScaler,
    model: Optional[LinearRegression],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = node_df[FEATURES].values
    if len(values) < (window + horizon):
        return np.empty((0, len(FEATURES))), np.empty((0, len(FEATURES))), np.array([], dtype=int)

    scaled = scaler.transform(values)
    X, y_true, target_idx = create_sequences(scaled, window, horizon)
    if len(X) == 0:
        return np.empty((0, len(FEATURES))), np.empty((0, len(FEATURES))), np.array([], dtype=int)

    if model_name == "linear":
        if model is None:
            return np.empty((0, len(FEATURES))), np.empty((0, len(FEATURES))), np.array([], dtype=int)
        y_pred = model.predict(X)
        y_pred = np.clip(y_pred, 0.0, 1.0)
    elif model_name == "naive_last":
        y_pred = X.reshape(len(X), window, len(FEATURES))[:, -1, :]
    elif model_name == "naive_mean":
        y_pred = X.reshape(len(X), window, len(FEATURES)).mean(axis=1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return y_true, y_pred, target_idx


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8), axis=0
    ) * 100

    return {
        "num_predictions": int(len(y_true)),
        "mae_per_feature": {f: float(v) for f, v in zip(FEATURES, mae)},
        "rmse_per_feature": {f: float(v) for f, v in zip(FEATURES, rmse)},
        "smape_per_feature": {f: float(v) for f, v in zip(FEATURES, smape)},
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(rmse.mean()),
        "smape_mean": float(smape.mean()),
    }


def aggregate_metrics(y_true_blocks: List[np.ndarray], y_pred_blocks: List[np.ndarray]) -> Optional[Dict]:
    if not y_true_blocks:
        return None
    y_true = np.concatenate(y_true_blocks, axis=0)
    y_pred = np.concatenate(y_pred_blocks, axis=0)
    return metric_bundle(y_true, y_pred)


def ci95(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, mean, mean
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    margin = 1.96 * sem
    return mean, mean - margin, mean + margin


def evaluate_run_set(
    data_by_run_node: Dict[str, Dict[str, pd.DataFrame]],
    train_runs: List[str],
    eval_runs: List[str],
    window: int,
    horizon: int,
    split_name: str,
    split_index: int,
) -> Tuple[List[Dict], List[Dict]]:
    fold_rows: List[Dict] = []
    phase_rows: List[Dict] = []

    all_nodes = sorted({node for r in eval_runs for node in data_by_run_node[r].keys()})
    node_models: Dict[str, Tuple[Optional[LinearRegression], Optional[MinMaxScaler]]] = {}

    for node in all_nodes:
        train_frames = [data_by_run_node[r][node] for r in train_runs if node in data_by_run_node[r]]
        model, scaler = train_linear_model(train_frames, window, horizon)
        node_models[node] = (model, scaler)

    for model_name in MODELS:
        y_true_all: List[np.ndarray] = []
        y_pred_all: List[np.ndarray] = []
        y_true_by_phase: Dict[str, List[np.ndarray]] = {}
        y_pred_by_phase: Dict[str, List[np.ndarray]] = {}

        for run_id in eval_runs:
            run_nodes = data_by_run_node[run_id]
            for node, node_df in run_nodes.items():
                model, scaler = node_models.get(node, (None, None))
                if scaler is None:
                    continue

                y_true, y_pred, target_idx = predict_with_model(model_name, node_df, window, horizon, scaler, model)
                if len(y_true) == 0:
                    continue

                y_true_all.append(y_true)
                y_pred_all.append(y_pred)

                phases = node_df.iloc[target_idx]["phase_name"].astype(str).to_numpy()
                for phase_name in sorted(set(phases.tolist())):
                    mask = phases == phase_name
                    if phase_name not in y_true_by_phase:
                        y_true_by_phase[phase_name] = []
                        y_pred_by_phase[phase_name] = []
                    y_true_by_phase[phase_name].append(y_true[mask])
                    y_pred_by_phase[phase_name].append(y_pred[mask])

        overall = aggregate_metrics(y_true_all, y_pred_all)
        if overall is None:
            continue

        fold_rows.append(
            {
                "split_name": split_name,
                "split_index": split_index,
                "window": window,
                "horizon": horizon,
                "model": model_name,
                "num_predictions": overall["num_predictions"],
                "mae_mean": overall["mae_mean"],
                "rmse_mean": overall["rmse_mean"],
                "smape_mean": overall["smape_mean"],
                "train_runs": "|".join(train_runs),
                "eval_runs": "|".join(eval_runs),
            }
        )

        for phase_name in sorted(y_true_by_phase.keys()):
            phase_metrics = aggregate_metrics(y_true_by_phase[phase_name], y_pred_by_phase[phase_name])
            if phase_metrics is None:
                continue
            phase_rows.append(
                {
                    "split_name": split_name,
                    "split_index": split_index,
                    "window": window,
                    "horizon": horizon,
                    "model": model_name,
                    "phase": phase_name,
                    "num_predictions": phase_metrics["num_predictions"],
                    "mae_mean": phase_metrics["mae_mean"],
                    "rmse_mean": phase_metrics["rmse_mean"],
                    "smape_mean": phase_metrics["smape_mean"],
                    "train_runs": "|".join(train_runs),
                    "eval_runs": "|".join(eval_runs),
                }
            )

    return fold_rows, phase_rows


def summarise_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, horizon, model_name), g in cv_df.groupby(["window", "horizon", "model"], sort=False):
        window_val = int(g["window"].iloc[0])
        horizon_val = int(g["horizon"].iloc[0])
        rmse_mean, rmse_lo, rmse_hi = ci95(g["rmse_mean"].tolist())
        mae_mean, mae_lo, mae_hi = ci95(g["mae_mean"].tolist())
        smape_mean, smape_lo, smape_hi = ci95(g["smape_mean"].tolist())
        rows.append(
            {
                "window": window_val,
                "horizon": horizon_val,
                "model": model_name,
                "folds": int(len(g)),
                "rmse_mean": rmse_mean,
                "rmse_ci_low": rmse_lo,
                "rmse_ci_high": rmse_hi,
                "mae_mean": mae_mean,
                "mae_ci_low": mae_lo,
                "mae_ci_high": mae_hi,
                "smape_mean": smape_mean,
                "smape_ci_low": smape_lo,
                "smape_ci_high": smape_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "rmse_mean", "mae_mean", "smape_mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 prediction validation (rolling-origin + strict holdout)")
    parser.add_argument("--runs-dir", required=True, help="Path to anomaly-detection/online-telemetry/dataset/runs")
    parser.add_argument("--output-dir", required=True, help="Directory for validation outputs")
    parser.add_argument("--run-ids", nargs="*", default=None, help="Optional subset of run ids to include")
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--min-train-runs", type=int, default=8, help="Minimum runs before first rolling fold")
    parser.add_argument("--holdout-count", type=int, default=4, help="Use latest N runs as strict holdout")
    parser.add_argument("--holdout-runs", nargs="*", default=None, help="Explicit run ids for strict holdout")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_by_run_node, ordered_runs, run_phase_sets = load_runs(args.runs_dir)
    if args.run_ids:
        requested = set(args.run_ids)
        ordered_runs = [run_id for run_id in ordered_runs if run_id in requested]
        data_by_run_node = {run_id: data_by_run_node[run_id] for run_id in ordered_runs}
        run_phase_sets = {run_id: run_phase_sets.get(run_id, set()) for run_id in ordered_runs}
        if not ordered_runs:
            raise RuntimeError("No valid runs remain after applying --run-ids filter")

    if args.holdout_runs:
        holdout_runs = [r for r in args.holdout_runs if r in ordered_runs]
        holdout_strategy = "explicit"
    else:
        holdout_runs = select_stratified_holdout(ordered_runs, run_phase_sets, args.holdout_count)
        holdout_strategy = "stratified_stress_control"

    dev_runs = [r for r in ordered_runs if r not in holdout_runs]

    if len(holdout_runs) < 1:
        raise RuntimeError("Need at least one strict holdout run")
    if len(dev_runs) < args.min_train_runs + 1:
        raise RuntimeError(
            f"Not enough development runs ({len(dev_runs)}) for rolling-origin with min-train-runs={args.min_train_runs}"
        )

    print(f"Loaded runs: {len(ordered_runs)}")
    print(f"Development runs: {len(dev_runs)} -> {dev_runs}")
    print(f"Strict holdout runs: {len(holdout_runs)} -> {holdout_runs}")
    print(f"Holdout strategy: {holdout_strategy}")

    cv_fold_rows: List[Dict] = []
    cv_phase_rows: List[Dict] = []

    split_count = 0
    for window in args.windows:
        for horizon in args.horizons:
            for split_idx in range(args.min_train_runs, len(dev_runs)):
                train_runs = dev_runs[:split_idx]
                val_run = [dev_runs[split_idx]]
                split_count += 1
                print(
                    f"[CV split {split_count}] window={window}, horizon={horizon}, "
                    f"train={len(train_runs)} runs, val={val_run[0]}"
                )
                fold_rows, phase_rows = evaluate_run_set(
                    data_by_run_node=data_by_run_node,
                    train_runs=train_runs,
                    eval_runs=val_run,
                    window=window,
                    horizon=horizon,
                    split_name="rolling_origin",
                    split_index=split_idx,
                )
                cv_fold_rows.extend(fold_rows)
                cv_phase_rows.extend(phase_rows)

    cv_df = pd.DataFrame(cv_fold_rows)
    cv_phase_df = pd.DataFrame(cv_phase_rows)
    if cv_df.empty:
        raise RuntimeError("No CV results generated")

    cv_summary = summarise_cv(cv_df)

    linear_rows = cv_summary[cv_summary["model"] == "linear"].sort_values("rmse_mean")
    if linear_rows.empty:
        raise RuntimeError("Linear model produced no CV summary rows")

    best_linear = linear_rows.iloc[0]
    best_window = int(best_linear["window"])
    best_horizon = int(best_linear["horizon"])

    print("\nSelected linear config from rolling-origin CV:")
    print(
        f"  window={best_window}, horizon={best_horizon}, "
        f"rmse_mean={best_linear['rmse_mean']:.4f}, mae_mean={best_linear['mae_mean']:.4f}"
    )

    holdout_fold_rows, holdout_phase_rows = evaluate_run_set(
        data_by_run_node=data_by_run_node,
        train_runs=dev_runs,
        eval_runs=holdout_runs,
        window=best_window,
        horizon=best_horizon,
        split_name="strict_holdout",
        split_index=0,
    )

    holdout_df = pd.DataFrame(holdout_fold_rows)
    holdout_phase_df = pd.DataFrame(holdout_phase_rows)

    outputs = {
        "cv_per_fold": os.path.join(args.output_dir, "cv_per_fold_metrics.csv"),
        "cv_per_fold_phase": os.path.join(args.output_dir, "cv_per_fold_phase_metrics.csv"),
        "cv_summary": os.path.join(args.output_dir, "cv_config_summary.csv"),
        "holdout_overall": os.path.join(args.output_dir, "holdout_overall_metrics.csv"),
        "holdout_phase": os.path.join(args.output_dir, "holdout_phase_metrics.csv"),
        "locked_config": os.path.join(args.output_dir, "locked_predictor_config.json"),
        "validation_summary": os.path.join(args.output_dir, "validation_summary.json"),
    }

    cv_df.to_csv(outputs["cv_per_fold"], index=False)
    cv_phase_df.to_csv(outputs["cv_per_fold_phase"], index=False)
    cv_summary.to_csv(outputs["cv_summary"], index=False)
    holdout_df.to_csv(outputs["holdout_overall"], index=False)
    holdout_phase_df.to_csv(outputs["holdout_phase"], index=False)

    holdout_focus = {}
    if not holdout_phase_df.empty:
        focus = holdout_phase_df[holdout_phase_df["phase"].isin(["baseline", "cpu", "mixed"])]
        for _, row in focus.iterrows():
            holdout_focus.setdefault(row["model"], {})[row["phase"]] = {
                "rmse_mean": float(row["rmse_mean"]),
                "mae_mean": float(row["mae_mean"]),
                "smape_mean": float(row["smape_mean"]),
                "num_predictions": int(row["num_predictions"]),
            }

    locked = {
        "selected_at": pd.Timestamp.now("UTC").isoformat(),
        "selection_strategy": "rolling_origin_cv_on_dev_runs",
        "holdout_strategy": holdout_strategy,
        "strict_holdout": holdout_runs,
        "development_runs": dev_runs,
        "config": {
            "model": "linear_regression",
            "window": best_window,
            "horizon": best_horizon,
            "scaling": "MinMaxScaler fit on training runs only",
        },
        "cv_linear_performance": {
            "rmse_mean": float(best_linear["rmse_mean"]),
            "rmse_ci_low": float(best_linear["rmse_ci_low"]),
            "rmse_ci_high": float(best_linear["rmse_ci_high"]),
            "mae_mean": float(best_linear["mae_mean"]),
            "smape_mean": float(best_linear["smape_mean"]),
            "folds": int(best_linear["folds"]),
        },
        "holdout_overall": {
            row["model"]: {
                "rmse_mean": float(row["rmse_mean"]),
                "mae_mean": float(row["mae_mean"]),
                "smape_mean": float(row["smape_mean"]),
                "num_predictions": int(row["num_predictions"]),
            }
            for _, row in holdout_df.iterrows()
        },
        "holdout_phase_focus": holdout_focus,
    }

    with open(outputs["locked_config"], "w") as f:
        json.dump(locked, f, indent=2)

    with open(outputs["validation_summary"], "w") as f:
        json.dump(
            {
                "runs_total": len(ordered_runs),
                "development_runs": len(dev_runs),
                "strict_holdout_runs": len(holdout_runs),
                "holdout_strategy": holdout_strategy,
                "windows": args.windows,
                "horizons": args.horizons,
                "rolling_origin_splits": int(cv_df["split_index"].nunique()),
                "best_linear_config": {"window": best_window, "horizon": best_horizon},
                "outputs": outputs,
            },
            f,
            indent=2,
        )

    print("\n=== Prediction Validation Complete ===")
    print(f"CV per-fold:          {outputs['cv_per_fold']}")
    print(f"CV phase metrics:     {outputs['cv_per_fold_phase']}")
    print(f"CV summary:           {outputs['cv_summary']}")
    print(f"Holdout overall:      {outputs['holdout_overall']}")
    print(f"Holdout phase:        {outputs['holdout_phase']}")
    print(f"Locked config:        {outputs['locked_config']}")
    print(f"Validation summary:   {outputs['validation_summary']}")


if __name__ == "__main__":
    main()
