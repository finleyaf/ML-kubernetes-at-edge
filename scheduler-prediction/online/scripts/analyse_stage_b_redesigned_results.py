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
    if "stress_" in name:
        return name.split("-", maxsplit=3)[-1]
    if "-stress-" in name:
        return "stress"
    if "-normal-" in name:
        return "normal"
    return "unknown"


def add_phase_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "phase_name" in out.columns:
        phase_name = out["phase_name"].fillna("").astype(str).str.strip()
        if "pod_name" in out.columns:
            inferred = out["pod_name"].map(infer_phase)
        else:
            inferred = pd.Series(["unknown"] * len(out), index=out.index)
        out["phase"] = phase_name.where(phase_name != "", inferred)
    elif "pod_name" in out.columns:
        out["phase"] = out["pod_name"].map(infer_phase)
    else:
        out["phase"] = "unknown"
    return out


def non_empty_string_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip() != ""


def maybe_float(value, digits: int = 4):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def maybe_percent(mask: pd.Series):
    if mask.empty:
        return None
    return round(float(mask.mean()) * 100.0, 2)


def jain_fairness_from_counts(node_counts: dict) -> float:
    if not node_counts:
        return 0.0
    values = [float(v) for v in node_counts.values()]
    n = float(len(values))
    denom = n * sum(v * v for v in values)
    if denom <= 0.0:
        return 0.0
    return (sum(values) ** 2) / denom


def capacity_alignment_from_counts(node_counts: dict, node_capacities: dict | None):
    if not node_counts or not node_capacities:
        return None

    relevant_nodes = [node for node in node_counts if node in node_capacities]
    if len(relevant_nodes) != len(node_counts):
        return None

    total = float(sum(node_counts.values()))
    if total <= 0.0:
        return None

    total_cpu = sum(float(node_capacities[node].get("cpu_millicores", 0.0)) for node in relevant_nodes)
    total_mem = sum(float(node_capacities[node].get("memory_mib", 0.0)) for node in relevant_nodes)
    if total_cpu <= 0.0 or total_mem <= 0.0:
        return None

    imbalance = 0.0
    for node in relevant_nodes:
        actual_share = float(node_counts[node]) / total
        cpu_share = float(node_capacities[node].get("cpu_millicores", 0.0)) / total_cpu
        mem_share = float(node_capacities[node].get("memory_mib", 0.0)) / total_mem
        expected_share = 0.5 * cpu_share + 0.5 * mem_share
        imbalance += abs(actual_share - expected_share)

    score = (1.0 - (0.5 * imbalance)) * 100.0
    return round(max(0.0, score), 4)


def pi_breakdown(df: pd.DataFrame, pi_node_name: str) -> dict:
    if "node" not in df.columns:
        return {
            "pi_node_name": pi_node_name,
            "pi_placements": None,
            "pi_placement_rate_percent": None,
            "non_pi_placements": None,
            "non_pi_placement_rate_percent": None,
        }

    node_series = df["node"].fillna("").astype(str).str.strip()
    pi_mask = node_series == pi_node_name
    non_empty_mask = node_series != ""
    non_pi_mask = (~pi_mask) & non_empty_mask

    return {
        "pi_node_name": pi_node_name,
        "pi_placements": int(pi_mask.sum()),
        "pi_placement_rate_percent": maybe_percent(pi_mask),
        "non_pi_placements": int(non_pi_mask.sum()),
        "non_pi_placement_rate_percent": maybe_percent(non_pi_mask),
    }


def summarize(df: pd.DataFrame, node_capacities: dict | None = None, pi_node_name: str = "raspberrypi") -> dict:
    node_counts = df["node"].value_counts().to_dict() if "node" in df.columns else {}
    pi_summary = pi_breakdown(df, pi_node_name)

    workloads = {}
    if "workload_type" in df.columns:
        for workload, subset in df.groupby("workload_type"):
            workload_summary = {
                "count": int(len(subset)),
                "avg_scheduling_latency_s": maybe_float(subset["scheduling_latency_s"].mean()),
                "avg_startup_time_s": maybe_float(subset["startup_time_s"].mean()),
                "avg_total_time_s": maybe_float(subset["total_time_s"].mean()),
                "avg_chosen_relative_cpu": maybe_float(subset["chosen_relative_cpu"].mean())
                if "chosen_relative_cpu" in subset.columns
                else None,
            }
            workload_summary.update(pi_breakdown(subset, pi_node_name))
            workloads[workload] = workload_summary

    summary = {
        "total_pods": int(len(df)),
        "node_placements": node_counts,
        "placement_fairness_percent": round(float(jain_fairness_from_counts(node_counts) * 100.0), 4),
        "capacity_alignment_percent": capacity_alignment_from_counts(node_counts, node_capacities),
        "avg_scheduling_latency_s": maybe_float(df["scheduling_latency_s"].mean()),
        "avg_startup_time_s": maybe_float(df["startup_time_s"].mean()),
        "avg_total_time_s": maybe_float(df["total_time_s"].mean()),
        "avg_chosen_relative_cpu": maybe_float(df["chosen_relative_cpu"].mean())
        if "chosen_relative_cpu" in df.columns
        else None,
        "high_contention_rate_percent": maybe_percent(df["high_contention"] == 1)
        if "high_contention" in df.columns
        else None,
        "pi_node_name": pi_summary["pi_node_name"],
        "pi_placements": pi_summary["pi_placements"],
        "pi_placement_rate_percent": pi_summary["pi_placement_rate_percent"],
        "non_pi_placements": pi_summary["non_pi_placements"],
        "non_pi_placement_rate_percent": pi_summary["non_pi_placement_rate_percent"],
        "workloads": workloads,
    }

    if "stress_target" in df.columns and "node" in df.columns:
        stress_subset = df[non_empty_string_mask(df["stress_target"])]
        if not stress_subset.empty:
            placed_on_target = stress_subset["node"] == stress_subset["stress_target"]
            placement_rate = maybe_percent(placed_on_target)
            summary["stress_target_placement_rate_percent"] = placement_rate
            summary["stress_target_avoidance_rate_percent"] = None if placement_rate is None else round(100.0 - placement_rate, 2)
        else:
            summary["stress_target_placement_rate_percent"] = None
            summary["stress_target_avoidance_rate_percent"] = None
    else:
        summary["stress_target_placement_rate_percent"] = None
        summary["stress_target_avoidance_rate_percent"] = None

    if "expected_node" in df.columns and "expected_node_match" in df.columns:
        expected_subset = df[non_empty_string_mask(df["expected_node"])]
        if not expected_subset.empty:
            summary["expected_node_match_rate_percent"] = maybe_percent(expected_subset["expected_node_match"] == 1)
        else:
            summary["expected_node_match_rate_percent"] = None
    else:
        summary["expected_node_match_rate_percent"] = None

    decision_fields = [
        ("avg_decision_total_score", "decision_total_score"),
        ("avg_decision_predicted_load", "decision_predicted_load"),
        ("avg_decision_base_predicted_load", "decision_base_predicted_load"),
        ("avg_decision_anomaly_risk", "decision_anomaly_risk"),
        ("avg_decision_cpu_request_fraction", "decision_cpu_request_fraction"),
        ("avg_decision_memory_request_fraction", "decision_memory_request_fraction"),
        ("avg_decision_capacity_penalty", "decision_capacity_penalty"),
    ]
    for summary_key, column in decision_fields:
        summary[summary_key] = maybe_float(df[column].mean()) if column in df.columns else None

    if "decision_prediction_source" in df.columns:
        counts = df["decision_prediction_source"].dropna().astype(str).str.strip()
        counts = counts[counts != ""]
        summary["decision_prediction_source_counts"] = counts.value_counts().to_dict()
    else:
        summary["decision_prediction_source_counts"] = {}

    return summary


def summarize_phases(df: pd.DataFrame, node_capacities: dict | None = None, pi_node_name: str = "raspberrypi") -> dict:
    phases = {}
    for phase, subset in df.groupby("phase"):
        phases[phase] = summarize(subset, node_capacities=node_capacities, pi_node_name=pi_node_name)
    return phases


def summarize_phase_workloads(df: pd.DataFrame, pi_node_name: str = "raspberrypi") -> dict:
    out = {}
    grouped = df.groupby(["phase", "workload_type"])
    for (phase, workload), subset in grouped:
        phase_workload_summary = {
            "count": int(len(subset)),
            "avg_scheduling_latency_s": maybe_float(subset["scheduling_latency_s"].mean()),
            "avg_startup_time_s": maybe_float(subset["startup_time_s"].mean()),
            "avg_total_time_s": maybe_float(subset["total_time_s"].mean()),
        }
        phase_workload_summary.update(pi_breakdown(subset, pi_node_name))
        out.setdefault(phase, {})[workload] = phase_workload_summary
    return out


def maybe_delta(custom: dict, baseline: dict, key: str, digits: int = 4):
    custom_value = custom.get(key)
    baseline_value = baseline.get(key)
    if custom_value is None or baseline_value is None:
        return None
    return round(float(custom_value) - float(baseline_value), digits)


def diff_simple(custom: dict, baseline: dict) -> dict:
    return {
        "placement_fairness_delta_pct": maybe_delta(custom, baseline, "placement_fairness_percent"),
        "capacity_alignment_delta_pct": maybe_delta(custom, baseline, "capacity_alignment_percent"),
        "scheduling_latency_delta_s": maybe_delta(custom, baseline, "avg_scheduling_latency_s"),
        "startup_time_delta_s": maybe_delta(custom, baseline, "avg_startup_time_s"),
        "total_time_delta_s": maybe_delta(custom, baseline, "avg_total_time_s"),
        "high_contention_rate_delta_pct": maybe_delta(custom, baseline, "high_contention_rate_percent", 2),
        "pi_placement_rate_delta_pct": maybe_delta(custom, baseline, "pi_placement_rate_percent", 2),
        "stress_target_placement_delta_pct": maybe_delta(custom, baseline, "stress_target_placement_rate_percent", 2),
        "stress_target_avoidance_delta_pct": maybe_delta(custom, baseline, "stress_target_avoidance_rate_percent", 2),
        "expected_node_match_delta_pct": maybe_delta(custom, baseline, "expected_node_match_rate_percent", 2),
        "chosen_relative_cpu_delta": maybe_delta(custom, baseline, "avg_chosen_relative_cpu"),
        "decision_total_score_delta": maybe_delta(custom, baseline, "avg_decision_total_score"),
        "decision_predicted_load_delta": maybe_delta(custom, baseline, "avg_decision_predicted_load"),
        "decision_anomaly_risk_delta": maybe_delta(custom, baseline, "avg_decision_anomaly_risk"),
        "decision_capacity_penalty_delta": maybe_delta(custom, baseline, "avg_decision_capacity_penalty"),
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
        "notes": [
            "utility components continue to use chosen_label/high_contention as secondary proxies",
            "primary redesigned online metrics are reported separately in the summary and phase views",
        ],
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
            "full utility components are available after rerun with the redesigned collector",
        ],
    }


def build_stage_b_utility_comparison(custom_components: dict, baseline_components: dict) -> dict:
    def component_delta(key: str):
        custom_value = custom_components["component_values"].get(key)
        baseline_value = baseline_components["component_values"].get(key)
        if custom_value is None or baseline_value is None:
            return None
        return round(float(custom_value) - float(baseline_value), 4)

    return {
        "utility_delta": round(
            float(custom_components["utility_score"]) - float(baseline_components["utility_score"]),
            6,
        ),
        "safe_placement_rate_delta": component_delta("safe_placement_rate"),
        "anomalous_rate_delta": component_delta("anomalous_rate"),
        "high_contention_rate_delta": component_delta("high_contention_decision_rate"),
        "avg_decision_latency_ms_delta": component_delta("avg_decision_latency_ms"),
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
                "scheduling_latency_delta_s": maybe_delta(
                    custom_pw[phase][workload],
                    baseline_pw[phase][workload],
                    "avg_scheduling_latency_s",
                ),
                "startup_time_delta_s": maybe_delta(
                    custom_pw[phase][workload],
                    baseline_pw[phase][workload],
                    "avg_startup_time_s",
                ),
                "total_time_delta_s": maybe_delta(
                    custom_pw[phase][workload],
                    baseline_pw[phase][workload],
                    "avg_total_time_s",
                ),
                "pi_placement_rate_delta_pct": maybe_delta(
                    custom_pw[phase][workload],
                    baseline_pw[phase][workload],
                    "pi_placement_rate_percent",
                    2,
                ),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize redesigned Stage B authoritative scheduler results")
    parser.add_argument("--baseline", required=True, help="CSV output from baseline arm")
    parser.add_argument("--custom", required=True, help="CSV output from custom arm")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--node-capacities", help="Optional JSON map of node allocatable capacity")
    parser.add_argument("--pi-node-name", default="raspberrypi", help="Node name treated as the Pi worker in heterogeneous runs")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    custom_path = Path(args.custom)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline CSV: {baseline_path}")
    if not custom_path.exists():
        raise FileNotFoundError(f"Missing custom CSV: {custom_path}")

    node_capacities = None
    if args.node_capacities:
        with open(args.node_capacities, "r", encoding="utf-8") as f:
            node_capacities = json.load(f)

    baseline_df = add_phase_column(pd.read_csv(baseline_path))
    custom_df = add_phase_column(pd.read_csv(custom_path))

    baseline_summary = summarize(baseline_df, node_capacities=node_capacities, pi_node_name=args.pi_node_name)
    custom_summary = summarize(custom_df, node_capacities=node_capacities, pi_node_name=args.pi_node_name)
    baseline_phases = summarize_phases(baseline_df, node_capacities=node_capacities, pi_node_name=args.pi_node_name)
    custom_phases = summarize_phases(custom_df, node_capacities=node_capacities, pi_node_name=args.pi_node_name)
    baseline_phase_workloads = summarize_phase_workloads(baseline_df, pi_node_name=args.pi_node_name)
    custom_phase_workloads = summarize_phase_workloads(custom_df, pi_node_name=args.pi_node_name)

    comparison = diff_simple(custom_summary, baseline_summary)
    has_full_cols = all(
        column in baseline_df.columns and column in custom_df.columns
        for column in ["chosen_label", "high_contention"]
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
            "node_capacities": args.node_capacities,
            "pi_node_name": args.pi_node_name,
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