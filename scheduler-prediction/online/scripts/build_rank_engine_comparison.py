#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    base_runs = root / "anomaly-detection/online-telemetry/dataset/runs"
    smoke_path = root / "scheduler-prediction/online/results/compare_rank_engine_smoke/latest_ranking.json"
    baseline_path = root / "scheduler-prediction/online/results/compare_rank_engine_baseline/baseline_analysis.json"
    plan_path = base_runs / "campaign_plan_compare_rank_engine_campaign.json"
    out_dir = root / "scheduler-prediction/online/results/compare_rank_engine_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in [smoke_path, baseline_path, plan_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required artifact: {required}")

    smoke = json.load(open(smoke_path, "r", encoding="utf-8"))
    baseline = json.load(open(baseline_path, "r", encoding="utf-8"))
    plan = json.load(open(plan_path, "r", encoding="utf-8"))

    run_rows = []
    recommended_node = smoke.get("best_node")

    for run in plan.get("runs", []):
        run_id = run.get("run_id")
        kind = run.get("kind", "unknown")
        labelled_path = base_runs / run_id / "labelled.csv"

        if not labelled_path.exists():
            run_rows.append(
                {
                    "run_id": run_id,
                    "kind": kind,
                    "rows": 0,
                    "normal_rows": 0,
                    "anomalous_rows": 0,
                    "anomaly_rate_pct": None,
                    "recommended_node_rows": 0,
                    "recommended_node_share_pct": None,
                    "recommended_node_anomaly_rate_pct": None,
                    "other_nodes_anomaly_rate_pct": None,
                }
            )
            continue

        df = pd.read_csv(labelled_path)
        if "label" not in df.columns or "node" not in df.columns:
            raise ValueError(f"label/node columns missing in {labelled_path}")

        rows = len(df)
        anom = int((df["label"] == 1).sum())
        normal = int((df["label"] == 0).sum())

        rec_df = df[df["node"] == recommended_node]
        other_df = df[df["node"] != recommended_node]

        rec_rows = len(rec_df)
        rec_anom_rate = float(rec_df["label"].mean() * 100.0) if rec_rows else None
        other_anom_rate = float(other_df["label"].mean() * 100.0) if len(other_df) else None

        run_rows.append(
            {
                "run_id": run_id,
                "kind": kind,
                "rows": int(rows),
                "normal_rows": normal,
                "anomalous_rows": anom,
                "anomaly_rate_pct": round((anom / rows) * 100.0, 3) if rows else None,
                "recommended_node_rows": int(rec_rows),
                "recommended_node_share_pct": round((rec_rows / rows) * 100.0, 3) if rows else None,
                "recommended_node_anomaly_rate_pct": round(rec_anom_rate, 3) if rec_anom_rate is not None else None,
                "other_nodes_anomaly_rate_pct": round(other_anom_rate, 3) if other_anom_rate is not None else None,
            }
        )

    runs_df = pd.DataFrame(run_rows).sort_values(["kind", "run_id"])
    runs_csv = out_dir / "campaign_run_comparison.csv"
    runs_df.to_csv(runs_csv, index=False)

    summary = {
        "inputs": {
            "smoke_ranking": smoke_path.as_posix(),
            "baseline_analysis": baseline_path.as_posix(),
            "campaign_plan": plan_path.as_posix(),
        },
        "rank_engine": {
            "best_node": recommended_node,
            "anomaly_source": smoke.get("anomaly_source"),
            "weights": smoke.get("weights", {}),
        },
        "baseline_default_scheduler": {
            "normal_avg_startup_time_s": baseline.get("normal", {}).get("avg_startup_time_s"),
            "stress_avg_startup_time_s": baseline.get("stress", {}).get("avg_startup_time_s"),
            "startup_time_increase_s": baseline.get("comparison", {}).get("startup_time_increase"),
            "stress_node_placement_pct": baseline.get("comparison", {}).get("stress_node_placement_pct"),
            "normal_node_placements": baseline.get("normal", {}).get("node_placements", {}),
            "stress_node_placements": baseline.get("stress", {}).get("node_placements", {}),
        },
        "campaign_rank_engine_data": {
            "planned_runs": int(len(plan.get("runs", []))),
            "evaluated_runs": int(runs_df["rows"].gt(0).sum()) if not runs_df.empty else 0,
            "total_rows": int(runs_df["rows"].sum()) if not runs_df.empty else 0,
            "total_anomalous_rows": int(runs_df["anomalous_rows"].sum()) if not runs_df.empty else 0,
            "overall_anomaly_rate_pct": round(
                float((runs_df["anomalous_rows"].sum() / max(1, runs_df["rows"].sum())) * 100.0), 3
            )
            if not runs_df.empty
            else None,
            "stress_runs_mean_anomaly_rate_pct": round(
                float(runs_df[runs_df["kind"] == "stress"]["anomaly_rate_pct"].dropna().mean()), 3
            )
            if not runs_df.empty and not runs_df[runs_df["kind"] == "stress"].empty
            else None,
            "control_runs_mean_anomaly_rate_pct": round(
                float(runs_df[runs_df["kind"] == "control"]["anomaly_rate_pct"].dropna().mean()), 3
            )
            if not runs_df.empty and not runs_df[runs_df["kind"] == "control"].empty
            else None,
            "recommended_node_share_pct_mean": round(float(runs_df["recommended_node_share_pct"].dropna().mean()), 3)
            if not runs_df.empty
            else None,
            "recommended_node_anomaly_rate_pct_mean": round(
                float(runs_df["recommended_node_anomaly_rate_pct"].dropna().mean()), 3
            )
            if not runs_df.empty
            else None,
            "other_nodes_anomaly_rate_pct_mean": round(
                float(runs_df["other_nodes_anomaly_rate_pct"].dropna().mean()), 3
            )
            if not runs_df.empty
            else None,
        },
        "interpretation_note": "This compares default scheduler baseline outcomes against rank-engine campaign telemetry statistics. It is not a strict A/B of node-placement policy because rank output was not wired as kube scheduler plugin/binder in this run.",
    }

    summary_json = out_dir / "comparison_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("saved", runs_csv.as_posix())
    print("saved", summary_json.as_posix())
    print("best_node", summary["rank_engine"]["best_node"])
    print("overall_anomaly_rate_pct", summary["campaign_rank_engine_data"]["overall_anomaly_rate_pct"])
    print("stress_runs_mean_anomaly_rate_pct", summary["campaign_rank_engine_data"]["stress_runs_mean_anomaly_rate_pct"])
    print("control_runs_mean_anomaly_rate_pct", summary["campaign_rank_engine_data"]["control_runs_mean_anomaly_rate_pct"])
    print("baseline_startup_increase_s", summary["baseline_default_scheduler"]["startup_time_increase_s"])
    print("baseline_stress_node_placement_pct", summary["baseline_default_scheduler"]["stress_node_placement_pct"])


if __name__ == "__main__":
    main()
