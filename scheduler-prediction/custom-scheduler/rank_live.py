import argparse
import json
import os

import pandas as pd

from hybrid_scheduler import HybridScheduler, FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with recent telemetry including node + feature columns")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--window", type=int, default=10, help="Prediction window size")
    parser.add_argument("--pred-weight", type=float, default=0.9, help="Weight for predicted load")
    parser.add_argument("--anomaly-weight", type=float, default=0.1, help="Weight for anomaly risk")
    parser.add_argument("--anomaly-history", type=int, default=45, help="Anomaly monitor history length")
    parser.add_argument("--anomaly-source", choices=["zscore", "nsa", "kmeans"], default="nsa", help="Anomaly detector source")
    parser.add_argument("--z-threshold", type=float, default=2.5, help="Anomaly z-score threshold")
    parser.add_argument("--nsa-num-detectors", type=int, default=120, help="NSA detector count")
    parser.add_argument("--nsa-radius", type=float, default=0.9, help="NSA detector radius")
    parser.add_argument("--kmeans-threshold-std", type=float, default=2.0, help="KMeans distance threshold multiplier")
    parser.add_argument("--output", help="Optional path to save ranking JSON")
    args = parser.parse_args()

    if abs((args.pred_weight + args.anomaly_weight) - 1.0) > 1e-6:
        raise ValueError("--pred-weight + --anomaly-weight must equal 1.0")

    df = pd.read_csv(args.input)
    workers = [n for n in sorted(df["node"].unique()) if "control" not in n]
    if not workers:
        raise ValueError("No worker nodes found in input data")

    scheduler = HybridScheduler(
        model_dir=args.model_dir,
        nodes=workers,
        window_size=args.window,
        anomaly_history=args.anomaly_history,
        anomaly_z_threshold=args.z_threshold,
        anomaly_source=args.anomaly_source,
        nsa_num_detectors=args.nsa_num_detectors,
        nsa_radius=args.nsa_radius,
        kmeans_threshold_std=args.kmeans_threshold_std,
        weight_prediction=args.pred_weight,
        weight_anomaly=args.anomaly_weight,
    )

    # Warm the scheduler with historical samples in timestamp order.
    df = df[df["node"].isin(workers)].sort_values("timestamp")

    latest_observation = {}
    for ts in sorted(df["timestamp"].unique()):
        tdf = df[df["timestamp"] == ts]
        for node in workers:
            ndf = tdf[tdf["node"] == node]
            if ndf.empty:
                continue
            row = ndf.iloc[0]
            obs = {f: float(row[f]) for f in FEATURES}
            scheduler.update(node, obs)
            latest_observation[node] = obs

    ranking = scheduler.score_nodes(latest_observation)
    if not ranking:
        raise RuntimeError("No ranking generated. Check input schema and node data coverage.")

    out = {
        "best_node": ranking[0].node,
        "ranking": [
            {
                "node": r.node,
                "total_score": r.total_score,
                "predicted_load": r.predicted_load,
                "anomaly_risk": r.anomaly_risk,
            }
            for r in ranking
        ],
        "weights": {
            "prediction": args.pred_weight,
            "anomaly": args.anomaly_weight,
        },
        "anomaly_source": args.anomaly_source,
    }

    print("\n=== Live Node Ranking ===")
    for i, r in enumerate(out["ranking"], start=1):
        print(
            f"{i}. {r['node']} | score={r['total_score']} "
            f"(pred={r['predicted_load']}, anomaly={r['anomaly_risk']})"
        )
    print(f"\nBest node: {out['best_node']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved ranking JSON to: {args.output}")


if __name__ == "__main__":
    main()
