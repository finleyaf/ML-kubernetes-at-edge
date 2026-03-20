import argparse
import json
import os
from typing import Dict, List

import pandas as pd

from hybrid_scheduler import HybridScheduler, PredictionOnlyScheduler, FEATURES


def build_observation(row: pd.Series) -> Dict[str, float]:
    return {f: float(row[f]) for f in FEATURES}


def summarise(results: List[Dict]) -> Dict:
    if not results:
        return {
            "decisions": 0,
            "anomalous_placements": 0,
            "safe_placements": 0,
            "safe_placement_rate": 0.0,
            "avg_total_score": 0.0,
            "avg_prediction_score": 0.0,
            "avg_anomaly_score": 0.0,
        }

    safe = sum(1 for r in results if r["chosen_label"] == 0)
    anomalous = len(results) - safe

    return {
        "decisions": len(results),
        "anomalous_placements": anomalous,
        "safe_placements": safe,
        "safe_placement_rate": round(100.0 * safe / len(results), 2),
        "avg_total_score": round(sum(r["total_score"] for r in results) / len(results), 4),
        "avg_prediction_score": round(sum(r["predicted_load"] for r in results) / len(results), 4),
        "avg_anomaly_score": round(sum(r["anomaly_risk"] for r in results) / len(results), 4),
    }


def simulate(df: pd.DataFrame, scheduler, nodes: List[str], warmup_steps: int) -> List[Dict]:
    decisions = []

    by_ts = sorted(df["timestamp"].unique())
    for step, ts in enumerate(by_ts):
        tdf = df[df["timestamp"] == ts]

        observations_by_node = {}
        labels_by_node = {}

        for node in nodes:
            ndf = tdf[tdf["node"] == node]
            if ndf.empty:
                continue
            row = ndf.iloc[0]
            observations_by_node[node] = build_observation(row)
            labels_by_node[node] = int(row["label"])

        if len(observations_by_node) != len(nodes):
            # Skip timestamps where any worker sample is missing.
            continue

        # Update model state first so decision uses latest telemetry.
        for node, obs in observations_by_node.items():
            scheduler.update(node, obs)

        if step < warmup_steps:
            continue

        chosen = scheduler.choose_node(observations_by_node)
        if chosen is None:
            continue

        decisions.append(
            {
                "timestamp": int(ts),
                "chosen_node": chosen.node,
                "chosen_label": labels_by_node[chosen.node],
                "total_score": chosen.total_score,
                "predicted_load": chosen.predicted_load,
                "anomaly_risk": chosen.anomaly_risk,
                "node_labels": labels_by_node,
            }
        )

    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to labelled telemetry CSV")
    parser.add_argument("--model-dir", required=True, help="Path to trained predictor models")
    parser.add_argument("--output", required=True, help="Path to save simulation summary JSON")
    parser.add_argument("--window", type=int, default=10, help="Prediction window size")
    parser.add_argument("--warmup", type=int, default=15, help="Initial timesteps skipped before decisions")
    parser.add_argument("--pred-weight", type=float, default=0.6, help="Hybrid weight for predicted load")
    parser.add_argument("--anomaly-weight", type=float, default=0.4, help="Hybrid weight for anomaly risk")
    parser.add_argument("--anomaly-history", type=int, default=30, help="Rolling history size for anomaly monitor")
    parser.add_argument("--z-threshold", type=float, default=2.5, help="Z-score threshold mapped to anomaly risk")
    args = parser.parse_args()

    if abs((args.pred_weight + args.anomaly_weight) - 1.0) > 1e-6:
        raise ValueError("--pred-weight + --anomaly-weight must equal 1.0")

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    workers = [n for n in sorted(df["node"].unique()) if "control" not in n]
    if not workers:
        raise ValueError("No worker nodes found in dataset")

    print(f"Workers: {workers}")

    hybrid = HybridScheduler(
        model_dir=args.model_dir,
        nodes=workers,
        window_size=args.window,
        anomaly_history=args.anomaly_history,
        anomaly_z_threshold=args.z_threshold,
        weight_prediction=args.pred_weight,
        weight_anomaly=args.anomaly_weight,
    )
    pred_only = PredictionOnlyScheduler(
        model_dir=args.model_dir,
        nodes=workers,
        window_size=args.window,
    )

    hybrid_decisions = simulate(df, hybrid, workers, warmup_steps=args.warmup)
    pred_only_decisions = simulate(df, pred_only, workers, warmup_steps=args.warmup)

    hybrid_summary = summarise(hybrid_decisions)
    pred_only_summary = summarise(pred_only_decisions)

    comparison = {
        "safe_placement_rate_delta": round(
            hybrid_summary["safe_placement_rate"] - pred_only_summary["safe_placement_rate"],
            2,
        ),
        "anomalous_placements_delta": (
            hybrid_summary["anomalous_placements"] - pred_only_summary["anomalous_placements"]
        ),
    }

    output = {
        "config": {
            "window": args.window,
            "warmup": args.warmup,
            "pred_weight": args.pred_weight,
            "anomaly_weight": args.anomaly_weight,
            "anomaly_history": args.anomaly_history,
            "z_threshold": args.z_threshold,
            "workers": workers,
        },
        "hybrid_scheduler": hybrid_summary,
        "prediction_only_scheduler": pred_only_summary,
        "comparison": comparison,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Custom Scheduler Simulation ===")
    print(f"Hybrid safe placement rate:      {hybrid_summary['safe_placement_rate']}%")
    print(f"Prediction-only safe rate:       {pred_only_summary['safe_placement_rate']}%")
    print(f"Delta (hybrid - pred-only):      {comparison['safe_placement_rate_delta']}%")
    print(f"Hybrid anomalous placements:     {hybrid_summary['anomalous_placements']}")
    print(f"Prediction-only anomalous count: {pred_only_summary['anomalous_placements']}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
