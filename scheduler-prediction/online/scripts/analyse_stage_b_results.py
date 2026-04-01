#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_UTILITY_WEIGHTS = {
    "safe_placement_rate": 0.45,
    "anomalous_rate": 0.25,
    "high_contention_decision_rate": 0.15,
    "latency": 0.10,
    "placement_fairness_percent": 0.05,
}

LATENCY_SCALE_MS = 1000.0


def infer_phase(pod_name: str) -> str:
    name = str(pod_name).lower()
    if "-stress-" in name:
        return "stress"
    if "-normal-" in name:
        return "normal"
    return "unknown"


def add_phase_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pod_name" in out.columns:
        out["phase"] = out["pod_name"].map(infer_phase)
    else:
        out["phase"] = "unknown"
    return out


def summarize(df: pd.DataFrame) -> dict:
    node_counts = df["node"].value_counts().to_dict() if "node" in df.columns else {}

    workloads = {}
    if "workload_type" in df.columns:
        for workload, subset in df.groupby("workload_type"):
            workloads[workload] = {
                "count": int(len(subset)),
                "avg_scheduling_latency_s": round(float(subset["scheduling_latency_s"].mean()), 4),
                "avg_startup_time_s": round(float(subset["startup_time_s"].mean()), 4),
                "avg_total_time_s": round(float(subset["total_time_s"].mean()), 4),
            }

    return {
        "total_pods": int(len(df)),
        "node_placements": node_counts,
        "placement_fairness_percent": round(float(jain_fairness_from_counts(node_counts) * 100.0), 4),
        "avg_scheduling_latency_s": round(float(df["scheduling_latency_s"].mean()), 4),
        "avg_startup_time_s": round(float(df["startup_time_s"].mean()), 4),
        "avg_total_time_s": round(float(df["total_time_s"].mean()), 4),
        "workloads": workloads,
    }


def jain_fairness_from_counts(node_counts: dict) -> float:
    if not node_counts:
        return 0.0
    values = [float(v) for v in node_counts.values()]
    n = float(len(values))
    denom = n * sum(v * v for v in values)
    if denom <= 0.0:
        return 0.0
    return (sum(values) ** 2) / denom


def summarize_phases(df: pd.DataFrame) -> dict:
    phases = {}
    for phase, subset in df.groupby("phase"):
        phases[phase] = summarize(subset)
    return phases


def summarize_phase_workloads(df: pd.DataFrame) -> dict:
    out = {}
    grouped = df.groupby(["phase", "workload_type"])
    for (phase, workload), subset in grouped:
        out.setdefault(phase, {})[workload] = {
            "count": int(len(subset)),
            "avg_scheduling_latency_s": round(float(subset["scheduling_latency_s"].mean()), 4),
            "avg_startup_time_s": round(float(subset["startup_time_s"].mean()), 4),
            "avg_total_time_s": round(float(subset["total_time_s"].mean()), 4),
        }
    return out


def diff_simple(custom: dict, baseline: dict) -> dict:
    return {
        "placement_fairness_delta_pct": round(
            custom["placement_fairness_percent"] - baseline["placement_fairness_percent"], 4
        ),
        "scheduling_latency_delta_s": round(custom["avg_scheduling_latency_s"] - baseline["avg_scheduling_latency_s"], 4),
        "startup_time_delta_s": round(custom["avg_startup_time_s"] - baseline["avg_startup_time_s"], 4),
        "total_time_delta_s": round(custom["avg_total_time_s"] - baseline["avg_total_time_s"], 4),
    }


def build_stage_b_utility_components(df: pd.DataFrame, summary: dict, weights: dict) -> dict:
    total = max(1, int(len(df)))
    safe_placements = int((df["chosen_label"] == 0).sum()) if "chosen_label" in df.columns else 0
    anomalous_placements = int((df["chosen_label"] == 1).sum()) if "chosen_label" in df.columns else 0
    high_contention = int((df["high_contention"] == 1).sum()) if "high_contention" in df.columns else 0

    safe_rate = round((100.0 * safe_placements) / total, 2)
    anomalous_rate = round((100.0 * anomalous_placements) / total, 2)
    contention_rate = round((100.0 * high_contention) / total, 2)
    fairness_pct = round(float(summary.get("placement_fairness_percent", 0.0)), 4)
    latency_ms = round(float(summary.get("avg_scheduling_latency_s", 0.0)) * 1000.0, 4)

    w_safe = float(weights.get("safe_placement_rate", 0.45))
    w_anom = float(weights.get("anomalous_rate", 0.25))
    w_contention = float(weights.get("high_contention_decision_rate", 0.15))
    w_lat = float(weights.get("latency", 0.10))
    w_fair = float(weights.get("placement_fairness_percent", 0.05))

    safe_component = round(w_safe * safe_rate, 6)
    anomalous_penalty = round(-w_anom * anomalous_rate, 6)
    contention_penalty = round(-w_contention * contention_rate, 6)
    latency_penalty = round(-w_lat * (latency_ms / LATENCY_SCALE_MS), 6)
    fairness_component = round(w_fair * fairness_pct, 6)

    utility_score = round(
        safe_component + anomalous_penalty + contention_penalty + latency_penalty + fairness_component,
        6,
    )

    return {
        "available_components": {
            "safe_placement_rate": True,
            "anomalous_rate": True,
            "high_contention_decision_rate": True,
            "latency": True,
            "placement_fairness_percent": True,
        },
        "component_values": {
            "safe_placement_rate": safe_rate,
            "anomalous_rate": anomalous_rate,
            "high_contention_decision_rate": contention_rate,
            "avg_decision_latency_ms": latency_ms,
            "placement_fairness_percent": fairness_pct,
            "safe_placements": safe_placements,
            "anomalous_placements": anomalous_placements,
        },
        "weighted_components": {
            "safe_component": safe_component,
            "anomalous_penalty": anomalous_penalty,
            "contention_penalty": contention_penalty,
            "latency_penalty": latency_penalty,
            "fairness_component": fairness_component,
        },
        "utility_score": utility_score,
    }


def build_stage_b_utility_components_legacy(summary: dict, weights: dict) -> dict:
    fairness_pct = float(summary.get("placement_fairness_percent", 0.0))
    latency_ms = round(float(summary.get("avg_scheduling_latency_s", 0.0)) * 1000.0, 4)

    w_lat = float(weights.get("latency", 0.10))
    w_fair = float(weights.get("placement_fairness_percent", 0.05))

    latency_penalty = round(-w_lat * (latency_ms / LATENCY_SCALE_MS), 6)
    fairness_component = round(w_fair * fairness_pct, 6)
    utility_score = round(latency_penalty + fairness_component, 6)

    return {
        "available_components": {
            "safe_placement_rate": False,
            "anomalous_rate": False,
            "high_contention_decision_rate": False,
            "latency": True,
            "placement_fairness_percent": True,
        },
        "component_values": {
            "safe_placement_rate": None,
            "anomalous_rate": None,
            "high_contention_decision_rate": None,
            "avg_decision_latency_ms": latency_ms,
            "placement_fairness_percent": round(fairness_pct, 4),
            "safe_placements": None,
            "anomalous_placements": None,
        },
        "weighted_components": {
            "safe_component": None,
            "anomalous_penalty": None,
            "contention_penalty": None,
            "latency_penalty": latency_penalty,
            "fairness_component": fairness_component,
        },
        "utility_score": utility_score,
        "notes": [
            "legacy Stage B CSV lacks chosen_label/high_contention",
            "full utility components available after rerun with updated collector",
        ],
    }


def build_stage_b_utility_comparison(custom_components: dict, baseline_components: dict) -> dict:
    def maybe_delta(key: str):
        c = custom_components["component_values"].get(key)
        b = baseline_components["component_values"].get(key)
        if c is None or b is None:
            return None
        return round(float(c) - float(b), 4)

    return {
        "utility_delta": round(
            float(custom_components["utility_score"]) - float(baseline_components["utility_score"]), 6
        ),
        "safe_placement_rate_delta": maybe_delta("safe_placement_rate"),
        "anomalous_rate_delta": maybe_delta("anomalous_rate"),
        "high_contention_rate_delta": maybe_delta("high_contention_decision_rate"),
        "avg_decision_latency_ms_delta": maybe_delta("avg_decision_latency_ms"),
        "fairness_component_delta": round(
            float(custom_components["weighted_components"]["fairness_component"])
            - float(baseline_components["weighted_components"]["fairness_component"]),
            6,
        ),
        "latency_penalty_delta": round(
            float(custom_components["weighted_components"]["latency_penalty"])
            - float(baseline_components["weighted_components"]["latency_penalty"]),
            6,
        ),
    }


def phase_comparison(custom_phases: dict, baseline_phases: dict) -> dict:
    out = {}
    for phase in sorted(set(custom_phases.keys()) & set(baseline_phases.keys())):
        out[phase] = diff_simple(custom_phases[phase], baseline_phases[phase])
    return out


def phase_workload_comparison(custom_pw: dict, baseline_pw: dict) -> dict:
    out = {}
    for phase in sorted(set(custom_pw.keys()) & set(baseline_pw.keys())):
        out[phase] = {}
        for workload in sorted(set(custom_pw[phase].keys()) & set(baseline_pw[phase].keys())):
            out[phase][workload] = {
                "scheduling_latency_delta_s": round(
                    custom_pw[phase][workload]["avg_scheduling_latency_s"]
                    - baseline_pw[phase][workload]["avg_scheduling_latency_s"],
                    4,
                ),
                "startup_time_delta_s": round(
                    custom_pw[phase][workload]["avg_startup_time_s"]
                    - baseline_pw[phase][workload]["avg_startup_time_s"],
                    4,
                ),
                "total_time_delta_s": round(
                    custom_pw[phase][workload]["avg_total_time_s"]
                    - baseline_pw[phase][workload]["avg_total_time_s"],
                    4,
                ),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage B matched-arm scheduler results")
    parser.add_argument("--baseline", required=True, help="CSV output from baseline arm")
    parser.add_argument("--custom", required=True, help="CSV output from custom arm")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    custom_path = Path(args.custom)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {baseline_path}")
    if not custom_path.exists():
        raise FileNotFoundError(f"Missing custom CSV: {custom_path}")

    baseline_df = add_phase_column(pd.read_csv(baseline_path))
    custom_df = add_phase_column(pd.read_csv(custom_path))

    baseline_summary = summarize(baseline_df)
    custom_summary = summarize(custom_df)
    baseline_phases = summarize_phases(baseline_df)
    custom_phases = summarize_phases(custom_df)
    baseline_phase_workloads = summarize_phase_workloads(baseline_df)
    custom_phase_workloads = summarize_phase_workloads(custom_df)

    comparison = diff_simple(custom_summary, baseline_summary)
    has_full_cols = all(
        c in baseline_df.columns and c in custom_df.columns
        for c in ["chosen_label", "high_contention"]
    )
    if has_full_cols:
        baseline_components = build_stage_b_utility_components(baseline_df, baseline_summary, DEFAULT_UTILITY_WEIGHTS)
        custom_components = build_stage_b_utility_components(custom_df, custom_summary, DEFAULT_UTILITY_WEIGHTS)
    else:
        baseline_components = build_stage_b_utility_components_legacy(baseline_summary, DEFAULT_UTILITY_WEIGHTS)
        custom_components = build_stage_b_utility_components_legacy(custom_summary, DEFAULT_UTILITY_WEIGHTS)

    summary = {
        "inputs": {
            "baseline_csv": baseline_path.as_posix(),
            "custom_csv": custom_path.as_posix(),
        },
        "locked_utility_weights": DEFAULT_UTILITY_WEIGHTS,
        "latency_scale_ms": LATENCY_SCALE_MS,
        "baseline_arm": baseline_summary,
        "custom_arm": custom_summary,
        "baseline_arm_utility_components": baseline_components,
        "custom_arm_utility_components": custom_components,
        "baseline_arm_by_phase": baseline_phases,
        "custom_arm_by_phase": custom_phases,
        "baseline_arm_by_phase_workload": baseline_phase_workloads,
        "custom_arm_by_phase_workload": custom_phase_workloads,
        "comparison_custom_minus_baseline": comparison,
        "comparison_custom_minus_baseline_utility_components": build_stage_b_utility_comparison(
            custom_components,
            baseline_components,
        ),
        "comparison_custom_minus_baseline_by_phase": phase_comparison(custom_phases, baseline_phases),
        "comparison_custom_minus_baseline_by_phase_workload": phase_workload_comparison(
            custom_phase_workloads,
            baseline_phase_workloads,
        ),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"saved {output_path.as_posix()}")
    print(f"baseline startup {baseline_summary['avg_startup_time_s']}")
    print(f"custom startup {custom_summary['avg_startup_time_s']}")
    print(f"delta startup {comparison['startup_time_delta_s']}")


if __name__ == "__main__":
    main()
