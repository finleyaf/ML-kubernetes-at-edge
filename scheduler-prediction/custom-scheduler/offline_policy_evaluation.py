import argparse
import json
import os
import pickle
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from hybrid_scheduler import HybridScheduler, PredictionOnlyScheduler


FEATURES = ["cpu_user", "cpu_system", "ram_used", "net_received", "net_sent"]


@dataclass
class PolicyStats:
    decisions: int
    safe_placements: int
    anomalous_placements: int
    safe_placement_rate: float
    avg_decision_latency_ms: float
    p95_decision_latency_ms: float


class AnomalyOnlyScheduler(HybridScheduler):
    """Baseline ranking using anomaly risk only (no prediction term)."""

    def __init__(
        self,
        model_dir: str,
        nodes: List[str],
        window_size: int = 10,
        anomaly_history: int = 30,
        anomaly_z_threshold: float = 2.5,
        anomaly_source: str = "zscore",
        nsa_num_detectors: int = 120,
        nsa_radius: float = 0.9,
        kmeans_threshold_std: float = 2.0,
    ):
        super().__init__(
            model_dir=model_dir,
            nodes=nodes,
            window_size=window_size,
            anomaly_history=anomaly_history,
            anomaly_z_threshold=anomaly_z_threshold,
            anomaly_source=anomaly_source,
            nsa_num_detectors=nsa_num_detectors,
            nsa_radius=nsa_radius,
            kmeans_threshold_std=kmeans_threshold_std,
            weight_prediction=0.0,
            weight_anomaly=1.0,
        )


def parse_run_number(run_id: str) -> int:
    try:
        return int(run_id.split("_")[-1])
    except Exception:
        return 0


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_protocol(path: Optional[str]) -> Dict:
    defaults = {
        "version": 1,
        "scheduler": {
            "adaptive_weighting": False,
            "adaptive_risk_low": 0.2,
            "adaptive_risk_high": 0.7,
            "adaptive_max_shift": 0.35,
            "adaptive_min_prediction_weight": 0.05,
            "adaptive_max_prediction_weight": 0.95,
        },
        "selection": {
            "objective": "utility",
            "utility_weights": {
                "safe_placement_rate": 0.45,
                "anomalous_rate": 0.25,
                "high_contention_decision_rate": 0.15,
                "latency": 0.1,
                "placement_fairness_percent": 0.05,
            },
            "latency_scale_ms": 1.0,
            "contention_relative_cpu_threshold": 1.1,
            "tie_breakers": ["hybrid_safe_rate", "hybrid_anomalous", "hybrid_latency"],
        },
        "data_split": {
            "freeze_test_runs": True,
            "test_runs": [],
        },
        "consistency_checks": {
            "hybrid_vs_prediction_only": {
                "min_positive_safe_delta_fraction": 0.75,
                "require_nonpositive_anomalous_delta": True,
            }
        },
    }

    if not path or not os.path.exists(path):
        return defaults

    with open(path, "r") as f:
        user_cfg = json.load(f)

    # shallow merge of known sections
    merged = defaults.copy()
    for k in ["scheduler", "selection", "data_split", "consistency_checks"]:
        merged[k] = defaults[k].copy()
        merged[k].update(user_cfg.get(k, {}))

    merged["version"] = user_cfg.get("version", defaults["version"])
    return merged


def resolve_window_size(model_dir: str, requested_window: Optional[int]) -> int:
    if requested_window is not None:
        return int(requested_window)

    summary_path = os.path.join(model_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        if "window" in summary:
            return int(summary["window"])

    model_files = sorted(
        f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pkl")
    )
    if not model_files:
        raise RuntimeError(f"No model_*.pkl files found in {model_dir}")

    first_model = os.path.join(model_dir, model_files[0])
    with open(first_model, "rb") as f:
        model = pickle.load(f)

    n_features = int(getattr(model, "n_features_in_", 0))
    if n_features <= 0 or (n_features % len(FEATURES)) != 0:
        raise RuntimeError(
            "Unable to infer window size from model shape. "
            "Please pass --window explicitly."
        )
    return n_features // len(FEATURES)


def collect_all_run_ids(runs_dir: str) -> List[str]:
    runs = [
        d
        for d in os.listdir(runs_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(runs_dir, d))
    ]
    return sorted(runs, key=parse_run_number)


def resolve_splits(
    runs_dir: str,
    split_config: Optional[str],
    validation_runs: Optional[List[str]],
    test_runs: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    available = collect_all_run_ids(runs_dir)
    if not available:
        raise RuntimeError(f"No run_* directories found under {runs_dir}")

    if split_config and os.path.exists(split_config):
        with open(split_config, "r") as f:
            cfg = json.load(f)
        val = [r for r in cfg.get("development_runs", []) if r in available]
        test = [r for r in cfg.get("strict_holdout", []) if r in available]
        if val and test:
            return val, test

    if validation_runs and test_runs:
        val = [r for r in validation_runs if r in available]
        test = [r for r in test_runs if r in available]
        if val and test:
            return val, test

    split = max(1, int(len(available) * 0.8))
    val = available[:split]
    test = available[split:]
    if not test:
        test = available[-1:]
        val = available[:-1]
    return val, test


def load_run_frame(runs_dir: str, run_id: str) -> pd.DataFrame:
    p = os.path.join(runs_dir, run_id, "labelled.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing labelled file: {p}")

    df = pd.read_csv(p)
    needed = ["timestamp", "node", "label", *FEATURES]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"{run_id}: missing columns {missing}")

    workers = [n for n in sorted(df["node"].unique()) if "control" not in str(n).lower()]
    if not workers:
        raise RuntimeError(f"{run_id}: no worker nodes found")

    keep_cols = ["timestamp", "node", "label", *FEATURES]
    out = df[keep_cols].copy()
    out = out[out["node"].isin(workers)]
    out["run_id"] = run_id
    return out.sort_values(["timestamp", "node"]).reset_index(drop=True)


def build_observation(row: pd.Series) -> Dict[str, float]:
    return {f: float(row[f]) for f in FEATURES}


def compute_policy_stats(records: List[Dict]) -> PolicyStats:
    if not records:
        return PolicyStats(
            decisions=0,
            safe_placements=0,
            anomalous_placements=0,
            safe_placement_rate=0.0,
            avg_decision_latency_ms=0.0,
            p95_decision_latency_ms=0.0,
        )

    decisions = len(records)
    safe = sum(1 for r in records if r["chosen_label"] == 0)
    anomalous = decisions - safe
    lat = [float(r["decision_latency_ms"]) for r in records]
    lat_sorted = sorted(lat)
    p95_idx = min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))

    return PolicyStats(
        decisions=decisions,
        safe_placements=safe,
        anomalous_placements=anomalous,
        safe_placement_rate=round(100.0 * safe / decisions, 2),
        avg_decision_latency_ms=round(float(statistics.mean(lat)), 4),
        p95_decision_latency_ms=round(float(lat_sorted[p95_idx]), 4),
    )


def compute_load_balancing_metrics(records: List[Dict]) -> Dict:
    if not records:
        return {
            "placement_fairness_jain": 0.0,
            "placement_fairness_percent": 0.0,
        }

    counts: Dict[str, int] = {}
    for r in records:
        node = str(r["chosen_node"])
        counts[node] = counts.get(node, 0) + 1

    vals = [float(v) for v in counts.values()]
    denom = len(vals) * sum(v * v for v in vals)
    jain = 0.0 if denom <= 0 else (sum(vals) ** 2) / denom
    jain = max(0.0, min(1.0, jain))

    return {
        "placement_fairness_jain": round(jain, 6),
        "placement_fairness_percent": round(100.0 * jain, 2),
    }


def compute_contention_metrics(records: List[Dict], protocol: Dict) -> Dict:
    if not records:
        return {
            "high_contention_decision_rate": 0.0,
            "mean_relative_cpu_load": 0.0,
            "mean_chosen_observed_cpu": 0.0,
        }

    sel = protocol.get("selection", {})
    rel_cpu_threshold = float(sel.get("contention_relative_cpu_threshold", 1.1))

    rel_cpu = [float(r.get("chosen_relative_cpu", 0.0)) for r in records]
    chosen_cpu = [float(r.get("chosen_observed_cpu", 0.0)) for r in records]
    high_contention = sum(1 for v in rel_cpu if v >= rel_cpu_threshold)

    return {
        "high_contention_decision_rate": round(100.0 * high_contention / len(records), 2),
        "mean_relative_cpu_load": round(float(statistics.mean(rel_cpu)), 4),
        "mean_chosen_observed_cpu": round(float(statistics.mean(chosen_cpu)), 4),
    }


def compute_weight_behavior_metrics(records: List[Dict]) -> Dict:
    if not records:
        return {
            "avg_effective_prediction_weight": 0.0,
            "avg_effective_anomaly_weight": 0.0,
            "anomaly_weight_gt_prediction_fraction": 0.0,
            "near_anomaly_only_fraction": 0.0,
            "near_prediction_only_fraction": 0.0,
        }

    pred_w = [float(r.get("effective_pred_weight", 0.0)) for r in records]
    anom_w = [float(r.get("effective_anomaly_weight", 0.0)) for r in records]

    n = float(len(records))
    anom_gt_pred = sum(1 for pw, aw in zip(pred_w, anom_w) if aw > pw)
    near_anom_only = sum(1 for aw in anom_w if aw >= 0.9)
    near_pred_only = sum(1 for pw in pred_w if pw >= 0.9)

    return {
        "avg_effective_prediction_weight": round(float(statistics.mean(pred_w)), 4),
        "avg_effective_anomaly_weight": round(float(statistics.mean(anom_w)), 4),
        "anomaly_weight_gt_prediction_fraction": round(anom_gt_pred / n, 4),
        "near_anomaly_only_fraction": round(near_anom_only / n, 4),
        "near_prediction_only_fraction": round(near_pred_only / n, 4),
    }


def compute_utility(metrics: Dict, protocol: Dict) -> float:
    sel = protocol.get("selection", {})
    w = sel.get("utility_weights", {})

    w_safe = float(w.get("safe_placement_rate", 0.45))
    w_anom = float(w.get("anomalous_rate", 0.25))
    w_lat = float(w.get("latency", 0.1))
    w_contention = float(w.get("high_contention_decision_rate", 0.15))
    w_balance = float(w.get("placement_fairness_percent", 0.05))
    lat_scale = float(sel.get("latency_scale_ms", 1.0))
    lat_scale = max(lat_scale, 1e-9)

    safe_rate = float(metrics.get("safe_placement_rate", 0.0))
    anom_rate = float(metrics.get("anomalous_rate", 0.0))
    latency = float(metrics.get("avg_decision_latency_ms", 0.0))
    contention_rate = float(metrics.get("high_contention_decision_rate", 0.0))
    fairness_pct = float(metrics.get("placement_fairness_percent", 0.0))

    # Maximize utility: reward safety and balancing, penalize anomaly, contention, and latency.
    util = (
        (w_safe * safe_rate)
        - (w_anom * anom_rate)
        - (w_contention * contention_rate)
        - (w_lat * (latency / lat_scale))
        + (w_balance * fairness_pct)
    )
    return round(util, 6)


def replay_paired(
    run_df: pd.DataFrame,
    workers: List[str],
    warmup_steps: int,
    schedulers: Dict[str, HybridScheduler],
) -> Dict[str, List[Dict]]:
    per_policy: Dict[str, List[Dict]] = {k: [] for k in schedulers.keys()}
    timestamps = sorted(run_df["timestamp"].unique())

    for step, ts in enumerate(timestamps):
        tdf = run_df[run_df["timestamp"] == ts]

        observations_by_node: Dict[str, Dict[str, float]] = {}
        labels_by_node: Dict[str, int] = {}

        for node in workers:
            ndf = tdf[tdf["node"] == node]
            if ndf.empty:
                continue
            row = ndf.iloc[0]
            observations_by_node[node] = build_observation(row)
            labels_by_node[node] = int(row["label"])

        if len(observations_by_node) != len(workers):
            continue

        # Update all policies first so each policy sees identical inputs.
        for policy in schedulers.values():
            for node, obs in observations_by_node.items():
                policy.update(node, obs)

        if step < warmup_steps:
            continue

        chosen_by_policy = {}
        for name, policy in schedulers.items():
            start = time.perf_counter()
            chosen = policy.choose_node(observations_by_node)
            latency_ms = (time.perf_counter() - start) * 1000.0
            if chosen is None:
                chosen_by_policy[name] = None
                continue

            chosen_cpu = float(observations_by_node[chosen.node]["cpu_user"]) + float(
                observations_by_node[chosen.node]["cpu_system"]
            )
            all_cpu = [
                float(obs["cpu_user"]) + float(obs["cpu_system"])
                for obs in observations_by_node.values()
            ]
            mean_cpu = float(statistics.mean(all_cpu)) if all_cpu else 0.0
            rel_cpu = chosen_cpu / max(mean_cpu, 1e-9)

            chosen_by_policy[name] = {
                "chosen_node": chosen.node,
                "chosen_label": labels_by_node[chosen.node],
                "total_score": float(chosen.total_score),
                "predicted_load": float(chosen.predicted_load),
                "anomaly_risk": float(chosen.anomaly_risk),
                "effective_pred_weight": float(chosen.weight_prediction),
                "effective_anomaly_weight": float(chosen.weight_anomaly),
                "chosen_observed_cpu": chosen_cpu,
                "chosen_relative_cpu": rel_cpu,
                "decision_latency_ms": float(latency_ms),
            }

        # Strict paired comparison: include only points where all policies produced decisions.
        if any(v is None for v in chosen_by_policy.values()):
            continue

        for name, picked in chosen_by_policy.items():
            record = {
                "timestamp": int(ts),
                "run_id": str(run_df["run_id"].iloc[0]),
                **picked,
            }
            per_policy[name].append(record)

    return per_policy


def evaluate_split(
    runs_dir: str,
    run_ids: List[str],
    model_dir: str,
    window: int,
    warmup: int,
    anomaly_history: int,
    z_threshold: float,
    hybrid_pred_weight: float,
    protocol: Optional[Dict] = None,
    anomaly_source: str = "zscore",
    nsa_num_detectors: int = 120,
    nsa_radius: float = 0.9,
    kmeans_threshold_std: float = 2.0,
) -> Dict:
    protocol = protocol or load_protocol(None)

    sched_cfg = protocol.get("scheduler", {})

    all_records: Dict[str, List[Dict]] = {
        "prediction_only": [],
        "anomaly_only": [],
        "hybrid": [],
    }

    for run_id in run_ids:
        run_df = load_run_frame(runs_dir, run_id)
        workers = sorted(run_df["node"].unique())

        schedulers = {
            "prediction_only": PredictionOnlyScheduler(
                model_dir=model_dir,
                nodes=workers,
                window_size=window,
            ),
            "anomaly_only": AnomalyOnlyScheduler(
                model_dir=model_dir,
                nodes=workers,
                window_size=window,
                anomaly_history=anomaly_history,
                anomaly_z_threshold=z_threshold,
                anomaly_source=anomaly_source,
                nsa_num_detectors=nsa_num_detectors,
                nsa_radius=nsa_radius,
                kmeans_threshold_std=kmeans_threshold_std,
            ),
            "hybrid": HybridScheduler(
                model_dir=model_dir,
                nodes=workers,
                window_size=window,
                anomaly_history=anomaly_history,
                anomaly_z_threshold=z_threshold,
                anomaly_source=anomaly_source,
                nsa_num_detectors=nsa_num_detectors,
                nsa_radius=nsa_radius,
                kmeans_threshold_std=kmeans_threshold_std,
                weight_prediction=hybrid_pred_weight,
                weight_anomaly=1.0 - hybrid_pred_weight,
                adaptive_weighting=bool(sched_cfg.get("adaptive_weighting", False)),
                adaptive_risk_low=float(sched_cfg.get("adaptive_risk_low", 0.2)),
                adaptive_risk_high=float(sched_cfg.get("adaptive_risk_high", 0.7)),
                adaptive_max_shift=float(sched_cfg.get("adaptive_max_shift", 0.35)),
                adaptive_min_prediction_weight=float(
                    sched_cfg.get("adaptive_min_prediction_weight", 0.05)
                ),
                adaptive_max_prediction_weight=float(
                    sched_cfg.get("adaptive_max_prediction_weight", 0.95)
                ),
            ),
        }

        run_records = replay_paired(
            run_df=run_df,
            workers=workers,
            warmup_steps=warmup,
            schedulers=schedulers,
        )

        for name in all_records.keys():
            all_records[name].extend(run_records[name])

    stats = {name: compute_policy_stats(records) for name, records in all_records.items()}

    metrics = {}
    for name, stat in stats.items():
        anomalous_rate = 100.0 * stat.anomalous_placements / max(stat.decisions, 1)
        lb_metrics = compute_load_balancing_metrics(all_records[name])
        contention_metrics = compute_contention_metrics(all_records[name], protocol)
        weight_metrics = compute_weight_behavior_metrics(all_records[name])
        m = {
            **vars(stat),
            "anomalous_rate": round(anomalous_rate, 2),
            **lb_metrics,
            **contention_metrics,
            **weight_metrics,
        }
        m["utility_score"] = compute_utility(m, protocol)
        metrics[name] = m

    baseline = metrics["prediction_only"]
    comparisons = {
        "anomaly_only_vs_prediction_only": {
            "safe_placement_rate_delta": round(
                metrics["anomaly_only"]["safe_placement_rate"] - baseline["safe_placement_rate"], 2
            ),
            "anomalous_placements_delta": (
                metrics["anomaly_only"]["anomalous_placements"] - baseline["anomalous_placements"]
            ),
            "anomalous_rate_delta": round(
                metrics["anomaly_only"]["anomalous_rate"] - baseline["anomalous_rate"], 2
            ),
            "avg_latency_ms_delta": round(
                metrics["anomaly_only"]["avg_decision_latency_ms"] - baseline["avg_decision_latency_ms"], 4
            ),
            "high_contention_rate_delta": round(
                metrics["anomaly_only"]["high_contention_decision_rate"] - baseline["high_contention_decision_rate"], 2
            ),
            "placement_fairness_delta": round(
                metrics["anomaly_only"]["placement_fairness_percent"] - baseline["placement_fairness_percent"], 2
            ),
            "avg_effective_anomaly_weight_delta": round(
                metrics["anomaly_only"]["avg_effective_anomaly_weight"] - baseline["avg_effective_anomaly_weight"], 4
            ),
            "near_anomaly_only_fraction_delta": round(
                metrics["anomaly_only"]["near_anomaly_only_fraction"] - baseline["near_anomaly_only_fraction"], 4
            ),
            "utility_delta": round(
                metrics["anomaly_only"]["utility_score"] - baseline["utility_score"], 6
            ),
        },
        "hybrid_vs_prediction_only": {
            "safe_placement_rate_delta": round(
                metrics["hybrid"]["safe_placement_rate"] - baseline["safe_placement_rate"], 2
            ),
            "anomalous_placements_delta": (
                metrics["hybrid"]["anomalous_placements"] - baseline["anomalous_placements"]
            ),
            "anomalous_rate_delta": round(
                metrics["hybrid"]["anomalous_rate"] - baseline["anomalous_rate"], 2
            ),
            "avg_latency_ms_delta": round(
                metrics["hybrid"]["avg_decision_latency_ms"] - baseline["avg_decision_latency_ms"], 4
            ),
            "high_contention_rate_delta": round(
                metrics["hybrid"]["high_contention_decision_rate"] - baseline["high_contention_decision_rate"], 2
            ),
            "placement_fairness_delta": round(
                metrics["hybrid"]["placement_fairness_percent"] - baseline["placement_fairness_percent"], 2
            ),
            "avg_effective_anomaly_weight_delta": round(
                metrics["hybrid"]["avg_effective_anomaly_weight"] - baseline["avg_effective_anomaly_weight"], 4
            ),
            "near_anomaly_only_fraction_delta": round(
                metrics["hybrid"]["near_anomaly_only_fraction"] - baseline["near_anomaly_only_fraction"], 4
            ),
            "utility_delta": round(
                metrics["hybrid"]["utility_score"] - baseline["utility_score"], 6
            ),
        },
    }

    return {
        "runs": run_ids,
        "prediction_only": metrics["prediction_only"],
        "anomaly_only": metrics["anomaly_only"],
        "hybrid": metrics["hybrid"],
        "comparisons": comparisons,
    }


def choose_best_grid_result(grid_rows: List[Dict], protocol: Dict) -> Dict:
    if not grid_rows:
        raise RuntimeError("No grid results to select from")

    objective = protocol.get("selection", {}).get("objective", "utility")
    if objective == "safe_rate":
        # maximize safe rate, then lower anomaly count, contention, and latency
        return sorted(
            grid_rows,
            key=lambda r: (
                -r["validation"]["hybrid"]["safe_placement_rate"],
                r["validation"]["hybrid"]["anomalous_placements"],
                r["validation"]["hybrid"]["high_contention_decision_rate"],
                r["validation"]["hybrid"]["avg_decision_latency_ms"],
            ),
        )[0]

    # default objective: maximize utility, then safe rate, then anomaly count, then latency
    return sorted(
        grid_rows,
        key=lambda r: (
            -r["validation"]["hybrid"]["utility_score"],
            -r["validation"]["hybrid"]["safe_placement_rate"],
            r["validation"]["hybrid"]["anomalous_placements"],
            r["validation"]["hybrid"]["avg_decision_latency_ms"],
        ),
    )[0]


def evaluate_runwise_hybrid_consistency(
    runs_dir: str,
    run_ids: List[str],
    model_dir: str,
    window: int,
    warmup: int,
    anomaly_history: int,
    z_threshold: float,
    hybrid_pred_weight: float,
    protocol: Dict,
    anomaly_source: str = "zscore",
    nsa_num_detectors: int = 120,
    nsa_radius: float = 0.9,
    kmeans_threshold_std: float = 2.0,
) -> Dict:
    rows = []
    for run_id in run_ids:
        r = evaluate_split(
            runs_dir=runs_dir,
            run_ids=[run_id],
            model_dir=model_dir,
            window=window,
            warmup=warmup,
            anomaly_history=anomaly_history,
            z_threshold=z_threshold,
            hybrid_pred_weight=hybrid_pred_weight,
            anomaly_source=anomaly_source,
            nsa_num_detectors=nsa_num_detectors,
            nsa_radius=nsa_radius,
            kmeans_threshold_std=kmeans_threshold_std,
            protocol=protocol,
        )
        d = r["comparisons"]["hybrid_vs_prediction_only"]
        rows.append(
            {
                "run_id": run_id,
                "safe_placement_rate_delta": d["safe_placement_rate_delta"],
                "anomalous_placements_delta": d["anomalous_placements_delta"],
                "anomalous_rate_delta": d["anomalous_rate_delta"],
                "avg_latency_ms_delta": d["avg_latency_ms_delta"],
                "utility_delta": d["utility_delta"],
                "is_positive_safe_delta": d["safe_placement_rate_delta"] > 0,
                "is_nonpositive_anom_delta": d["anomalous_placements_delta"] <= 0,
            }
        )

    pos_frac = 0.0
    if rows:
        pos_frac = sum(1 for x in rows if x["is_positive_safe_delta"]) / len(rows)

    checks = protocol.get("consistency_checks", {}).get("hybrid_vs_prediction_only", {})
    min_pos = float(checks.get("min_positive_safe_delta_fraction", 0.75))
    require_nonpos_anom = bool(checks.get("require_nonpositive_anomalous_delta", True))

    passed = pos_frac >= min_pos
    if require_nonpos_anom:
        passed = passed and all(x["is_nonpositive_anom_delta"] for x in rows)

    return {
        "per_run": rows,
        "summary": {
            "positive_safe_delta_fraction": round(pos_frac, 4),
            "required_positive_safe_delta_fraction": min_pos,
            "require_nonpositive_anomalous_delta": require_nonpos_anom,
            "passed": bool(passed),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline paired evaluation for prediction/anomaly/hybrid policies")
    parser.add_argument("--runs-dir", required=True, help="Path to anomaly-detection/online-telemetry/dataset/runs")
    parser.add_argument("--model-dir", required=True, help="Path to trained predictor models")
    parser.add_argument("--output", required=True, help="Path to JSON summary output")
    parser.add_argument("--protocol", default="custom-scheduler/evaluation_protocol.json", help="Path to locked evaluation protocol JSON")
    parser.add_argument("--split-config", default="scheduler-prediction/prediction/results/phase4_validation/locked_predictor_config.json")
    parser.add_argument("--validation-runs", nargs="*", default=None)
    parser.add_argument("--test-runs", nargs="*", default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--anomaly-history", type=int, default=45)
    parser.add_argument("--anomaly-source", choices=["zscore", "nsa", "kmeans"], default="zscore")
    parser.add_argument("--nsa-num-detectors", type=int, default=120)
    parser.add_argument("--nsa-radius", type=float, default=0.9)
    parser.add_argument("--kmeans-threshold-std", type=float, default=2.0)
    parser.add_argument("--weight-grid", default="0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--z-grid", default="2.0,2.5,3.0,3.5")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    window_size = resolve_window_size(args.model_dir, args.window)
    weight_grid = parse_float_list(args.weight_grid)
    z_grid = parse_float_list(args.z_grid)

    for w in weight_grid:
        if w < 0.0 or w > 1.0:
            raise ValueError("All weight-grid values must be in [0, 1]")

    validation_runs, test_runs = resolve_splits(
        runs_dir=args.runs_dir,
        split_config=args.split_config,
        validation_runs=args.validation_runs,
        test_runs=args.test_runs,
    )

    if not validation_runs or not test_runs:
        raise RuntimeError("Both validation and test run sets must be non-empty")

    split_rules = protocol.get("data_split", {})
    if split_rules.get("freeze_test_runs", True):
        frozen_test = [r for r in split_rules.get("test_runs", []) if r]
        if frozen_test:
            available = set(collect_all_run_ids(args.runs_dir))
            frozen_test = [r for r in frozen_test if r in available]
            if not frozen_test:
                raise RuntimeError("Protocol freeze_test_runs is enabled but no protocol test runs were found")
            test_runs = sorted(frozen_test, key=parse_run_number)
            validation_runs = [r for r in validation_runs if r not in set(test_runs)]

    print(f"Validation runs ({len(validation_runs)}): {validation_runs}")
    print(f"Test runs ({len(test_runs)}): {test_runs}")
    print(f"Using window size: {window_size}")
    print(f"Using anomaly source: {args.anomaly_source}")
    print(f"Grid size: {len(weight_grid)} weights x {len(z_grid)} thresholds")

    grid_results: List[Dict] = []
    for pred_weight in weight_grid:
        for z_threshold in z_grid:
            print(f"Evaluating grid config: pred_weight={pred_weight}, z_threshold={z_threshold}")
            val_result = evaluate_split(
                runs_dir=args.runs_dir,
                run_ids=validation_runs,
                model_dir=args.model_dir,
                window=window_size,
                warmup=args.warmup,
                anomaly_history=args.anomaly_history,
                z_threshold=z_threshold,
                hybrid_pred_weight=pred_weight,
                anomaly_source=args.anomaly_source,
                nsa_num_detectors=args.nsa_num_detectors,
                nsa_radius=args.nsa_radius,
                kmeans_threshold_std=args.kmeans_threshold_std,
                protocol=protocol,
            )
            grid_results.append(
                {
                    "pred_weight": pred_weight,
                    "anomaly_weight": round(1.0 - pred_weight, 6),
                    "z_threshold": z_threshold,
                    "validation": val_result,
                }
            )

    best = choose_best_grid_result(grid_results, protocol)

    print(
        "Best validation config: "
        f"pred_weight={best['pred_weight']}, "
        f"anomaly_weight={best['anomaly_weight']}, "
        f"z_threshold={best['z_threshold']}"
    )

    test_result = evaluate_split(
        runs_dir=args.runs_dir,
        run_ids=test_runs,
        model_dir=args.model_dir,
        window=window_size,
        warmup=args.warmup,
        anomaly_history=args.anomaly_history,
        z_threshold=float(best["z_threshold"]),
        hybrid_pred_weight=float(best["pred_weight"]),
        anomaly_source=args.anomaly_source,
        nsa_num_detectors=args.nsa_num_detectors,
        nsa_radius=args.nsa_radius,
        kmeans_threshold_std=args.kmeans_threshold_std,
        protocol=protocol,
    )

    consistency_validation = evaluate_runwise_hybrid_consistency(
        runs_dir=args.runs_dir,
        run_ids=validation_runs,
        model_dir=args.model_dir,
        window=window_size,
        warmup=args.warmup,
        anomaly_history=args.anomaly_history,
        z_threshold=float(best["z_threshold"]),
        hybrid_pred_weight=float(best["pred_weight"]),
        anomaly_source=args.anomaly_source,
        nsa_num_detectors=args.nsa_num_detectors,
        nsa_radius=args.nsa_radius,
        kmeans_threshold_std=args.kmeans_threshold_std,
        protocol=protocol,
    )
    consistency_test = evaluate_runwise_hybrid_consistency(
        runs_dir=args.runs_dir,
        run_ids=test_runs,
        model_dir=args.model_dir,
        window=window_size,
        warmup=args.warmup,
        anomaly_history=args.anomaly_history,
        z_threshold=float(best["z_threshold"]),
        hybrid_pred_weight=float(best["pred_weight"]),
        anomaly_source=args.anomaly_source,
        nsa_num_detectors=args.nsa_num_detectors,
        nsa_radius=args.nsa_radius,
        kmeans_threshold_std=args.kmeans_threshold_std,
        protocol=protocol,
    )

    split_manifest = {
        "frozen_by_protocol": bool(split_rules.get("freeze_test_runs", True)),
        "protocol_path": args.protocol,
        "split_config_source": args.split_config,
        "validation_runs": validation_runs,
        "test_runs": test_runs,
    }

    output = {
        "config": {
            "window": window_size,
            "warmup": args.warmup,
            "anomaly_history": args.anomaly_history,
            "anomaly_source": args.anomaly_source,
            "nsa_num_detectors": args.nsa_num_detectors,
            "nsa_radius": args.nsa_radius,
            "kmeans_threshold_std": args.kmeans_threshold_std,
            "protocol_path": args.protocol,
            "validation_runs": validation_runs,
            "test_runs": test_runs,
            "weight_grid": weight_grid,
            "z_grid": z_grid,
        },
        "protocol": protocol,
        "split_manifest": split_manifest,
        "selection_rule": protocol.get("selection", {}),
        "grid_search_results": grid_results,
        "best_validation_config": {
            "pred_weight": best["pred_weight"],
            "anomaly_weight": best["anomaly_weight"],
            "z_threshold": best["z_threshold"],
            "validation_summary": best["validation"],
        },
        "consistency_validation": consistency_validation,
        "consistency_test": consistency_test,
        "untouched_test_evaluation": test_result,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    manifest_path = f"{os.path.splitext(args.output)[0]}_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(split_manifest, f, indent=2)

    print("\n=== Offline Policy Evaluation Complete ===")
    print(f"Saved: {args.output}")
    print(f"Split manifest: {manifest_path}")
    print(
        "Test hybrid vs prediction-only delta: "
        f"safe_rate={test_result['comparisons']['hybrid_vs_prediction_only']['safe_placement_rate_delta']}%, "
        f"anomalous={test_result['comparisons']['hybrid_vs_prediction_only']['anomalous_placements_delta']}, "
        f"latency_ms={test_result['comparisons']['hybrid_vs_prediction_only']['avg_latency_ms_delta']}"
    )


if __name__ == "__main__":
    main()
