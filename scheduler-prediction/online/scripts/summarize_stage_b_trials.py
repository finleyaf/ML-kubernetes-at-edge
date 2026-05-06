#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_summary(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage B trial summaries")
    parser.add_argument(
        "--results-dir",
        default="scheduler-prediction/online/results",
        help="Directory containing stage_b_authoritative_run_* outputs",
    )
    parser.add_argument(
        "--prefix",
        default="stage_b_authoritative_run_",
        help="Run directory prefix",
    )
    parser.add_argument(
        "--output",
        default="scheduler-prediction/online/results/stage_b_trials_aggregate.json",
        help="Aggregate output JSON",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = sorted([p for p in results_dir.glob(f"{args.prefix}*") if p.is_dir()])

    runs = []
    for run_dir in run_dirs:
        summary_path = run_dir / "stage_b_summary.json"
        if not summary_path.exists():
            continue
        data = load_summary(summary_path)
        cmp_all = data.get("comparison_custom_minus_baseline", {})
        cmp_phase = data.get("comparison_custom_minus_baseline_by_phase", {})
        cmp_pw = data.get("comparison_custom_minus_baseline_by_phase_workload", {})
        cmp_util = data.get("comparison_custom_minus_baseline_utility_components", {})
        runs.append(
            {
                "run_id": run_dir.name,
                "summary_path": summary_path.as_posix(),
                "overall": cmp_all,
                "utility": cmp_util,
                "by_phase": cmp_phase,
                "by_phase_workload": cmp_pw,
            }
        )

    if not runs:
        raise RuntimeError("No stage_b_summary.json files found")

    overall = {
        "scheduling_latency_delta_s_mean": mean([r["overall"].get("scheduling_latency_delta_s") for r in runs]),
        "startup_time_delta_s_mean": mean([r["overall"].get("startup_time_delta_s") for r in runs]),
        "total_time_delta_s_mean": mean([r["overall"].get("total_time_delta_s") for r in runs]),
    }

    by_phase = {}
    for phase in ["normal", "stress"]:
        by_phase[phase] = {
            "scheduling_latency_delta_s_mean": mean(
                [r["by_phase"].get(phase, {}).get("scheduling_latency_delta_s") for r in runs]
            ),
            "startup_time_delta_s_mean": mean(
                [r["by_phase"].get(phase, {}).get("startup_time_delta_s") for r in runs]
            ),
            "total_time_delta_s_mean": mean(
                [r["by_phase"].get(phase, {}).get("total_time_delta_s") for r in runs]
            ),
        }

    workloads = ["cpu", "memory", "mixed"]
    by_phase_workload = {}
    for phase in ["normal", "stress"]:
        by_phase_workload[phase] = {}
        for workload in workloads:
            by_phase_workload[phase][workload] = {
                "scheduling_latency_delta_s_mean": mean(
                    [
                        r["by_phase_workload"].get(phase, {}).get(workload, {}).get("scheduling_latency_delta_s")
                        for r in runs
                    ]
                ),
                "startup_time_delta_s_mean": mean(
                    [
                        r["by_phase_workload"].get(phase, {}).get(workload, {}).get("startup_time_delta_s")
                        for r in runs
                    ]
                ),
                "total_time_delta_s_mean": mean(
                    [
                        r["by_phase_workload"].get(phase, {}).get(workload, {}).get("total_time_delta_s")
                        for r in runs
                    ]
                ),
            }

    utility = {
        "utility_delta_mean": mean([r["utility"].get("utility_delta") for r in runs]),
        "safe_placement_rate_delta_mean": mean([r["utility"].get("safe_placement_rate_delta") for r in runs]),
        "anomalous_rate_delta_mean": mean([r["utility"].get("anomalous_rate_delta") for r in runs]),
        "high_contention_rate_delta_mean": mean([r["utility"].get("high_contention_rate_delta") for r in runs]),
        "avg_decision_latency_ms_delta_mean": mean([r["utility"].get("avg_decision_latency_ms_delta") for r in runs]),
    }

    # Identify strongest startup-time regression bucket
    worst_bucket = {"phase": None, "workload": None, "startup_time_delta_s_mean": -10**9}
    for phase, workloads_map in by_phase_workload.items():
        for workload, deltas in workloads_map.items():
            val = deltas.get("startup_time_delta_s_mean")
            if val is None:
                continue
            if val > worst_bucket["startup_time_delta_s_mean"]:
                worst_bucket = {
                    "phase": phase,
                    "workload": workload,
                    "startup_time_delta_s_mean": val,
                }

    aggregate = {
        "n_trials": len(runs),
        "trial_ids": [r["run_id"] for r in runs],
        "overall_custom_minus_baseline_mean": overall,
        "utility_components_custom_minus_baseline_mean": utility,
        "by_phase_custom_minus_baseline_mean": by_phase,
        "by_phase_workload_custom_minus_baseline_mean": by_phase_workload,
        "largest_startup_regression_bucket": worst_bucket,
        "per_trial": runs,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"saved {output.as_posix()}")
    print(f"n_trials {aggregate['n_trials']}")
    print(f"overall_startup_delta_mean {overall['startup_time_delta_s_mean']}")
    print(
        "worst_startup_bucket",
        worst_bucket["phase"],
        worst_bucket["workload"],
        worst_bucket["startup_time_delta_s_mean"],
    )


if __name__ == "__main__":
    main()
