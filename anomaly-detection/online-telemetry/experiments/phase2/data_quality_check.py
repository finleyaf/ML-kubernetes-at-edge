import argparse
import json
import os
from typing import Dict, List

import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "node",
    "cpu_user",
    "cpu_system",
    "cpu_iowait",
    "ram_used",
    "net_received",
    "net_sent",
    "load1",
    "label",
]


def load_run_metadata(run_dir: str) -> Dict:
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def load_run_phases(run_dir: str) -> List[Dict]:
    phases_path = os.path.join(run_dir, "phases.json")
    if not os.path.exists(phases_path):
        return []
    with open(phases_path, "r") as f:
        return json.load(f)


def per_run_checks(run_dir: str, expected_nodes: List[str], min_rows: int, min_anomaly_ratio: float) -> Dict:
    run_id = os.path.basename(run_dir)
    labelled_path = os.path.join(run_dir, "labelled.csv")
    phases = load_run_phases(run_dir)
    metadata = load_run_metadata(run_dir)

    result = {
        "run_id": run_id,
        "status": "pass",
        "issues": [],
        "kind": metadata.get("kind", "unknown"),
        "rows": 0,
        "anomaly_rows": 0,
        "anomaly_ratio": 0.0,
        "node_counts": {},
        "timestamp_min": None,
        "timestamp_max": None,
        "timestamp_span_s": 0,
        "missing_columns": [],
        "null_cells": 0,
        "phase_count": len(phases),
    }

    if not os.path.exists(labelled_path):
        result["status"] = "fail"
        result["issues"].append("missing labelled.csv")
        return result

    df = pd.read_csv(labelled_path)
    result["rows"] = int(len(df))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    result["missing_columns"] = missing_cols
    if missing_cols:
        result["status"] = "fail"
        result["issues"].append(f"missing required columns: {missing_cols}")
        return result

    result["null_cells"] = int(df[REQUIRED_COLUMNS].isnull().sum().sum())
    if result["null_cells"] > 0:
        result["status"] = "fail"
        result["issues"].append(f"found null values in required fields: {result['null_cells']}")

    if result["rows"] < min_rows:
        result["status"] = "fail"
        result["issues"].append(f"row count below threshold: {result['rows']} < {min_rows}")

    counts = df["node"].value_counts().to_dict()
    result["node_counts"] = {k: int(v) for k, v in counts.items()}

    missing_nodes = [n for n in expected_nodes if n not in counts]
    if missing_nodes:
        result["status"] = "fail"
        result["issues"].append(f"missing expected nodes: {missing_nodes}")

    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        if min_count == 0 or (max_count - min_count) > 2:
            result["status"] = "fail"
            result["issues"].append(
                f"node sampling imbalance too high: max={max_count}, min={min_count}"
            )

    result["anomaly_rows"] = int((df["label"] == 1).sum())
    result["anomaly_ratio"] = round(result["anomaly_rows"] / max(1, result["rows"]), 4)

    kind = result["kind"]
    if kind == "control":
        if result["anomaly_rows"] != 0:
            result["status"] = "fail"
            result["issues"].append("control run contains anomalous labels")
    elif kind == "stress":
        if result["anomaly_ratio"] < min_anomaly_ratio:
            result["status"] = "warn" if result["status"] == "pass" else result["status"]
            result["issues"].append(
                f"low anomaly ratio for stress run: {result['anomaly_ratio']} < {min_anomaly_ratio}"
            )

    ts_min = int(df["timestamp"].min())
    ts_max = int(df["timestamp"].max())
    result["timestamp_min"] = ts_min
    result["timestamp_max"] = ts_max
    result["timestamp_span_s"] = int(ts_max - ts_min)

    unique_ts = df["timestamp"].nunique()
    expected_rows_from_ts = unique_ts * len(expected_nodes)
    if abs(result["rows"] - expected_rows_from_ts) > len(expected_nodes):
        result["status"] = "warn" if result["status"] == "pass" else result["status"]
        result["issues"].append(
            "possible dropped samples: rows not close to unique_timestamps * node_count"
        )

    return result


def aggregate(results: List[Dict]) -> Dict:
    by_status = {"pass": 0, "warn": 0, "fail": 0}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    total_rows = sum(r["rows"] for r in results)
    total_anom = sum(r["anomaly_rows"] for r in results)
    stress = [r for r in results if r["kind"] == "stress"]
    control = [r for r in results if r["kind"] == "control"]

    return {
        "runs_total": len(results),
        "status_counts": by_status,
        "rows_total": int(total_rows),
        "anomaly_rows_total": int(total_anom),
        "anomaly_ratio_total": round(total_anom / max(1, total_rows), 4),
        "stress_runs": len(stress),
        "control_runs": len(control),
        "stress_avg_anomaly_ratio": round(
            sum(r["anomaly_ratio"] for r in stress) / max(1, len(stress)), 4
        ),
        "control_anomaly_rows_total": int(sum(r["anomaly_rows"] for r in control)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True, help="Path to dataset/runs directory")
    parser.add_argument("--output-json", required=True, help="Output JSON report path")
    parser.add_argument("--output-csv", required=True, help="Output CSV summary path")
    parser.add_argument(
        "--expected-nodes",
        nargs="+",
        default=["k3s-control", "k3s-worker-2", "k3s-worker-3"],
        help="Expected node names in each run",
    )
    parser.add_argument("--min-rows", type=int, default=250, help="Minimum acceptable rows per run")
    parser.add_argument(
        "--min-anomaly-ratio",
        type=float,
        default=0.05,
        help="Minimum anomaly ratio expected for stress runs",
    )
    args = parser.parse_args()

    run_dirs = sorted(
        [
            os.path.join(args.runs_dir, d)
            for d in os.listdir(args.runs_dir)
            if d.startswith("run_") and os.path.isdir(os.path.join(args.runs_dir, d))
        ]
    )

    results = [
        per_run_checks(
            run_dir=run_dir,
            expected_nodes=args.expected_nodes,
            min_rows=args.min_rows,
            min_anomaly_ratio=args.min_anomaly_ratio,
        )
        for run_dir in run_dirs
    ]

    summary = aggregate(results)

    report = {
        "config": {
            "runs_dir": os.path.abspath(args.runs_dir),
            "expected_nodes": args.expected_nodes,
            "min_rows": args.min_rows,
            "min_anomaly_ratio": args.min_anomaly_ratio,
        },
        "summary": summary,
        "runs": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)

    rows = []
    for r in results:
        rows.append(
            {
                "run_id": r["run_id"],
                "kind": r["kind"],
                "status": r["status"],
                "rows": r["rows"],
                "anomaly_rows": r["anomaly_rows"],
                "anomaly_ratio": r["anomaly_ratio"],
                "timestamp_span_s": r["timestamp_span_s"],
                "null_cells": r["null_cells"],
                "issues": " | ".join(r["issues"]),
            }
        )

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    print("=== Data Quality Check ===")
    print(f"Runs checked: {summary['runs_total']}")
    print(f"Status counts: {summary['status_counts']}")
    print(f"Total rows: {summary['rows_total']}")
    print(f"Total anomaly rows: {summary['anomaly_rows_total']}")
    print(f"Overall anomaly ratio: {summary['anomaly_ratio_total']}")
    print(f"Stress avg anomaly ratio: {summary['stress_avg_anomaly_ratio']}")
    print(f"Control anomaly rows total: {summary['control_anomaly_rows_total']}")
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
