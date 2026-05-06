#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

DEFAULT_WORKER_TARGETS = [
    "k3s-worker-2",
    "k3s-worker-3",
    "k3s-worker-4",
    "raspberrypi",
]


# Keep the offline audit path aligned with the final live Stage B policy family.
LOCKED_OFFLINE_WARMUP = 24
LOCKED_OFFLINE_ANOMALY_HISTORY = 45
LOCKED_OFFLINE_ANOMALY_SOURCE = "nsa"
LOCKED_OFFLINE_NSA_NUM_DETECTORS = 120
LOCKED_OFFLINE_NSA_RADIUS = 0.9
LOCKED_OFFLINE_PRED_WEIGHT = 0.9
LOCKED_OFFLINE_Z_THRESHOLD = 2.5


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def check_netdata_endpoint(base_url: str, timeout_s: int = 3) -> Tuple[bool, str]:
    probe_url = f"{base_url.rstrip('/')}/api/v1/info"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout_s) as resp:
            if 200 <= resp.status < 300:
                return True, probe_url
            return False, f"Unexpected status {resp.status} from {probe_url}"
    except urllib.error.URLError as exc:
        return False, f"{probe_url} unreachable: {exc}"
    except Exception as exc:
        return False, f"{probe_url} check failed: {exc}"


def parse_run_number(run_id: str) -> int:
    try:
        return int(run_id.split("_")[-1])
    except Exception:
        return 0


def list_run_ids(runs_dir: Path) -> List[str]:
    runs = [p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(runs, key=parse_run_number)


def next_run_ids(runs_dir: Path, count: int) -> List[str]:
    existing = list_run_ids(runs_dir)
    start = parse_run_number(existing[-1]) + 1 if existing else 1
    return [f"run_{i:03d}" for i in range(start, start + count)]


def stress_command(name: str, intensity: str, duration: int) -> str:
    if name == "cpu":
        cpu = {"low": 1, "medium": 2, "high": 3}[intensity]
        return f"stress --cpu {cpu} --timeout {duration}"
    if name == "io":
        io = {"low": 2, "medium": 4, "high": 6}[intensity]
        return f"stress --io {io} --timeout {duration}"
    if name == "memory":
        vm, bytes_ = {
            "low": (1, "512M"),
            "medium": (1, "1G"),
            "high": (2, "1G"),
        }[intensity]
        return f"stress --vm {vm} --vm-bytes {bytes_} --timeout {duration}"
    if name == "mixed":
        spec = {
            "low": "--cpu 1 --vm 1 --vm-bytes 512M",
            "medium": "--cpu 2 --vm 1 --vm-bytes 1G",
            "high": "--cpu 2 --vm 2 --vm-bytes 1G",
        }[intensity]
        return f"stress {spec} --timeout {duration}"
    raise ValueError(f"Unsupported stress type: {name}")


def target_for_event(worker_targets: List[str], pattern_idx: int, event_idx: int) -> str:
    if not worker_targets:
        raise ValueError("worker_targets must contain at least one worker node")
    return worker_targets[(pattern_idx + event_idx) % len(worker_targets)]


def make_novel_phases(
    baseline: int,
    recovery: int,
    durations: List[int],
    pattern_idx: int,
    worker_targets: List[str],
    safe_stress_profile: bool = False,
) -> List[Dict]:
    # Deliberately abrupt transitions and mixed patterns for novelty.
    patterns = [
        [
            ("cpu", "high"),
            ("memory", "low"),
            ("mixed", "high"),
            ("io", "medium"),
            ("cpu", "low"),
        ],
        [
            ("io", "high"),
            ("cpu", "medium"),
            ("memory", "high"),
            ("mixed", "medium"),
            ("io", "low"),
        ],
        [
            ("mixed", "high"),
            ("cpu", "high"),
            ("io", "low"),
            ("memory", "medium"),
            ("mixed", "low"),
        ],
    ]
    seq = patterns[pattern_idx % len(patterns)]

    phases = [{"type": "baseline", "name": "baseline", "duration": baseline}]
    for i, (name, intensity) in enumerate(seq):
        if safe_stress_profile and intensity == "high" and name in {"mixed", "memory"}:
            intensity = "medium"
        dur = durations[(pattern_idx + i) % len(durations)]
        phase_name = f"{name}_novel_{i+1}"
        phases.append(
            {
                "type": "stress",
                "name": phase_name,
                "target": target_for_event(worker_targets, pattern_idx, i),
                "intensity": intensity,
                "duration": int(dur),
                "command": stress_command(name, intensity, int(dur)),
            }
        )
    phases.append({"type": "recovery", "name": "recovery", "duration": recovery})
    return phases


def write_novel_plan(
    plan_path: Path,
    run_ids: List[str],
    baseline: int,
    recovery: int,
    durations: List[int],
    worker_targets: List[str],
    safe_stress_profile: bool = False,
) -> None:
    runs = []
    now = datetime.now(timezone.utc).isoformat()
    for idx, run_id in enumerate(run_ids, start=1):
        runs.append(
            {
                "run_id": run_id,
                "kind": "stress",
                "created_at": now,
                "phases": make_novel_phases(
                    baseline,
                    recovery,
                    durations,
                    idx - 1,
                    worker_targets,
                    safe_stress_profile=safe_stress_profile,
                ),
                "order": idx,
            }
        )

    plan = {
        "meta": {
            "generated_at": now,
            "generator": "run_additional_novel_pipeline.py",
            "runs": len(run_ids),
            "baseline": baseline,
            "recovery": recovery,
            "durations": durations,
            "worker_targets": worker_targets,
            "safe_stress_profile": safe_stress_profile,
        },
        "runs": runs,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


def assign_phase_name(ts: float, phases: List[Dict]) -> str:
    for idx, phase in enumerate(phases):
        start = float(phase["start"])
        end = float(phase["end"])
        is_last = idx == len(phases) - 1
        in_range = (start <= ts <= end) if is_last else (start <= ts < end)
        if in_range:
            phase_type = str(phase.get("type", "")).strip().lower()
            phase_name = str(phase.get("name", phase_type)).strip().lower()
            return phase_name if phase_name else (phase_type or "unknown")
    return "unknown"


def build_phase_labelled_dataset(runs_dir: Path, run_ids: List[str], out_csv: Path) -> Dict:
    frames = []
    phase_counts: Dict[str, int] = {}

    for run_id in run_ids:
        labelled = runs_dir / run_id / "labelled.csv"
        phases_json = runs_dir / run_id / "phases.json"
        if not labelled.exists() or not phases_json.exists():
            continue
        df = pd.read_csv(labelled)
        if "timestamp" not in df.columns:
            continue
        phases = json.load(open(phases_json, "r", encoding="utf-8"))
        df = df.copy()
        df["phase_name"] = df["timestamp"].apply(lambda ts: assign_phase_name(ts, phases))
        df["run_id"] = run_id
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid labelled data found for selected development runs")

    all_df = pd.concat(frames, ignore_index=True)
    for k, v in all_df["phase_name"].value_counts().to_dict().items():
        phase_counts[str(k)] = int(v)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_csv, index=False)

    return {
        "rows": int(len(all_df)),
        "runs": int(len(sorted(set(all_df["run_id"].tolist())))),
        "phase_counts": phase_counts,
    }


def discover_run_nodes(runs_dir: Path, run_id: str) -> List[str]:
    labelled = runs_dir / run_id / "labelled.csv"
    if not labelled.exists():
        return []
    df = pd.read_csv(labelled, usecols=["node"])
    return sorted(str(node) for node in df["node"].dropna().astype(str).unique().tolist())


def collect_nodes_for_runs(runs_dir: Path, run_ids: List[str]) -> List[str]:
    nodes = set()
    for run_id in run_ids:
        nodes.update(discover_run_nodes(runs_dir, run_id))
    return sorted(nodes)


def find_runs_with_required_nodes(runs_dir: Path, required_nodes: List[str]) -> List[str]:
    required = set(required_nodes)
    matches = []
    for run_id in list_run_ids(runs_dir):
        nodes = set(discover_run_nodes(runs_dir, run_id))
        if required.issubset(nodes):
            matches.append(run_id)
    return matches


def trainer_model_name(selection_model: str, fallback_model: str) -> str:
    mapping = {
        "linear_regression": "linear",
        "linear": "linear",
        "ridge": "ridge",
    }
    selected = mapping.get(str(selection_model).strip().lower())
    if selected:
        return selected
    return mapping.get(str(fallback_model).strip().lower(), "ridge")


def load_offline_eval_module(offline_eval_path: Path):
    spec = importlib.util.spec_from_file_location("offline_policy_evaluation", str(offline_eval_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {offline_eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect novel runs, retrain predictor, and run protected selection/audit")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--novel-runs", type=int, default=4)
    parser.add_argument("--durations", default="45,60,90", help="Comma-separated stress durations in seconds")
    parser.add_argument("--baseline", type=int, default=120)
    parser.add_argument("--recovery", type=int, default=120)
    parser.add_argument("--collect", action="store_true", help="Execute run_campaign.py using generated novel plan")
    parser.add_argument("--zone", default="europe-west2-c")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--ssh-retries", type=int, default=4)
    parser.add_argument("--ssh-retry-delay", type=int, default=20)
    parser.add_argument("--collector-startup-timeout", type=int, default=25)
    parser.add_argument("--stress-executor", choices=["ssh", "k8s-pod"], default="k8s-pod")
    parser.add_argument("--netdata-base-url", default="http://localhost:20000")
    parser.add_argument("--worker-targets", nargs="+", default=DEFAULT_WORKER_TARGETS)
    parser.add_argument("--refresh-validation-split", action="store_true")
    parser.add_argument("--validation-min-train-runs", type=int, default=8)
    parser.add_argument("--validation-holdout-count", type=int, default=4)
    parser.add_argument("--safe-stress-profile", action="store_true")
    parser.add_argument("--existing-plan", help="Use an existing campaign plan JSON instead of generating a new one")
    parser.add_argument("--output-tag", default="expanded_novel")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    runs_dir = root / "anomaly-detection" / "online-telemetry" / "dataset" / "runs"
    campaign_script = root / "anomaly-detection" / "online-telemetry" / "experiments" / "phase1" / "run_campaign.py"
    quality_script = root / "anomaly-detection" / "online-telemetry" / "experiments" / "phase2" / "data_quality_check.py"

    locked_cfg_path = root / "scheduler-prediction" / "prediction" / "results" / "phase4_validation" / "locked_predictor_config.json"
    base_model_dir = root / "scheduler-prediction" / "prediction" / "models" / "phase4_weighted_calibrated"
    train_script = root / "scheduler-prediction" / "prediction" / "train_predictor.py"
    validate_script = root / "scheduler-prediction" / "prediction" / "validate_prediction.py"
    calibration_script = root / "scheduler-prediction" / "prediction" / "phase_calibration_report.py"

    eval_script = root / "scheduler-prediction" / "custom-scheduler" / "offline_policy_evaluation.py"
    eval_protocol = root / "scheduler-prediction" / "custom-scheduler" / "evaluation_protocol.json"

    results_dir = root / "scheduler-prediction" / "custom-scheduler" / "results"
    pred_results_dir = root / "scheduler-prediction" / "prediction" / "results" / "phase4_validation"
    pred_data_dir = root / "scheduler-prediction" / "prediction" / "data"
    pred_models_dir = root / "scheduler-prediction" / "prediction" / "models"

    durations = [int(x.strip()) for x in args.durations.split(",") if x.strip()]
    if not durations:
        raise ValueError("--durations must contain at least one value")

    if args.existing_plan:
        plan_path = Path(args.existing_plan).resolve()
        with open(plan_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        novel_run_ids = [str(run.get("run_id")) for run in plan_data.get("runs", []) if run.get("run_id")]
        if not novel_run_ids:
            raise RuntimeError(f"No run IDs found in existing plan: {plan_path}")
        print(f"Using existing campaign plan: {plan_path}")
        print(f"Plan run IDs: {novel_run_ids}")
    else:
        novel_run_ids = next_run_ids(runs_dir, args.novel_runs)
        plan_path = runs_dir / f"campaign_plan_{args.output_tag}.json"
        write_novel_plan(
            plan_path,
            novel_run_ids,
            args.baseline,
            args.recovery,
            durations,
            args.worker_targets,
            safe_stress_profile=args.safe_stress_profile,
        )
        print(f"Generated novel campaign plan: {plan_path}")
        print(f"Planned run IDs: {novel_run_ids}")

    if args.collect:
        ok, netdata_msg = check_netdata_endpoint(args.netdata_base_url)
        if not ok:
            print(f"WARNING: Netdata preflight failed: {netdata_msg}")
            print("Collection will likely fail until Netdata endpoint is reachable.")
        else:
            print(f"Netdata preflight OK: {netdata_msg}")

        run_cmd(
            [
                sys.executable,
                str(campaign_script),
                "--plan",
                str(plan_path),
                "--zone",
                args.zone,
                "--base-url",
                args.netdata_base_url,
                "--interval",
                str(args.interval),
                "--stress-executor",
                args.stress_executor,
                "--ssh-retries",
                str(args.ssh_retries),
                "--ssh-retry-delay",
                str(args.ssh_retry_delay),
                "--collector-startup-timeout",
                str(args.collector_startup_timeout),
                "--nodes",
                "k3s-control",
                *args.worker_targets,
            ],
            cwd=root / "anomaly-detection" / "online-telemetry",
        )
    else:
        print("Collection skipped (--collect not set).")

    existing_runs = set(list_run_ids(runs_dir))
    collected_novel_runs = []
    for run_id in novel_run_ids:
        if run_id not in existing_runs:
            continue
        labelled_path = runs_dir / run_id / "labelled.csv"
        phases_path = runs_dir / run_id / "phases.json"
        if labelled_path.exists() and phases_path.exists():
            collected_novel_runs.append(run_id)
    if not collected_novel_runs:
        print("No successfully labelled novel runs found yet. Re-run with --collect to execute campaign.")
        print(f"Plan ready at: {plan_path}")
        return

    print(f"Collected novel runs detected: {collected_novel_runs}")

    required_nodes = ["k3s-control", *args.worker_targets]
    fair_cohort_runs = find_runs_with_required_nodes(runs_dir, required_nodes)
    print(f"Fair-cohort runs with complete node coverage: {fair_cohort_runs}")

    quality_json = pred_results_dir / f"data_quality_{args.output_tag}.json"
    quality_csv = pred_results_dir / f"data_quality_{args.output_tag}.csv"
    run_cmd(
        [
            sys.executable,
            str(quality_script),
            "--runs-dir",
            str(runs_dir),
            "--output-json",
            str(quality_json),
            "--output-csv",
            str(quality_csv),
            "--expected-nodes",
            *required_nodes,
            "--run-ids",
            *fair_cohort_runs,
        ],
        cwd=root,
    )

    min_required_fair_runs = args.validation_min_train_runs + args.validation_holdout_count + 1
    if len(fair_cohort_runs) < min_required_fair_runs:
        print(
            "Fair cohort is not yet large enough to rebuild the governed split: "
            f"need at least {min_required_fair_runs} runs with nodes {required_nodes}, "
            f"found {len(fair_cohort_runs)}."
        )
        print(f"Quality report: {quality_json}")
        print(f"Plan ready at: {plan_path}")
        return

    locked_cfg = json.load(open(locked_cfg_path, "r", encoding="utf-8"))
    locked_nodes = collect_nodes_for_runs(
        runs_dir,
        list(locked_cfg.get("development_runs", [])) + list(locked_cfg.get("strict_holdout", [])),
    )
    fair_mode = args.refresh_validation_split or not set(args.worker_targets).issubset(set(locked_nodes))

    if fair_mode:
        validation_output_dir = pred_results_dir / f"phase4_validation_{args.output_tag}"
        run_cmd(
            [
                sys.executable,
                str(validate_script),
                "--runs-dir",
                str(runs_dir),
                "--output-dir",
                str(validation_output_dir),
                "--run-ids",
                *fair_cohort_runs,
                "--min-train-runs",
                str(args.validation_min_train_runs),
                "--holdout-count",
                str(args.validation_holdout_count),
            ],
            cwd=root,
        )
        expanded_cfg_path = validation_output_dir / "locked_predictor_config.json"
        locked_cfg = json.load(open(expanded_cfg_path, "r", encoding="utf-8"))
        print(f"Rebuilt fair locked split config: {expanded_cfg_path}")
        strict_holdout = list(locked_cfg["strict_holdout"])
        dev_runs = list(locked_cfg["development_runs"])
    else:
        strict_holdout = list(locked_cfg["strict_holdout"])
        dev_runs = [r for r in locked_cfg["development_runs"] if r not in strict_holdout]
        for r in collected_novel_runs:
            if r not in strict_holdout and r not in dev_runs:
                dev_runs.append(r)
        dev_runs = sorted(dev_runs, key=parse_run_number)

        expanded_cfg = dict(locked_cfg)
        expanded_cfg["selected_at"] = datetime.now(timezone.utc).isoformat()
        expanded_cfg["selection_strategy"] = "locked_holdout_plus_novel_runs_added_to_dev"
        expanded_cfg["development_runs"] = dev_runs
        expanded_cfg_path = pred_results_dir / f"locked_predictor_config_{args.output_tag}.json"
        expanded_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(expanded_cfg_path, "w", encoding="utf-8") as f:
            json.dump(expanded_cfg, f, indent=2)
        print(f"Wrote expanded split config: {expanded_cfg_path}")

    dev_csv = pred_data_dir / f"dev_runs_phase_labelled_{args.output_tag}.csv"
    ds_meta = build_phase_labelled_dataset(runs_dir, dev_runs, dev_csv)
    print(f"Built expanded development dataset: {dev_csv}")
    print(f"Dataset stats: {ds_meta}")
    holdout_csv = pred_data_dir / f"holdout_runs_phase_labelled_{args.output_tag}.csv"
    holdout_meta = build_phase_labelled_dataset(runs_dir, strict_holdout, holdout_csv)
    print(f"Built holdout dataset: {holdout_csv}")
    print(f"Holdout stats: {holdout_meta}")

    train_summary = json.load(open(base_model_dir / "training_summary.json", "r", encoding="utf-8"))
    model_out_dir = pred_models_dir / f"phase4_weighted_calibrated_{args.output_tag}"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    selected_config = locked_cfg.get("config", {})
    train_window = int(selected_config.get("window", train_summary.get("window", 5)))
    train_horizon = int(selected_config.get("horizon", train_summary.get("horizon", 3)))
    train_model = trainer_model_name(selected_config.get("model", ""), train_summary.get("model", "ridge"))

    run_cmd(
        [
            sys.executable,
            str(train_script),
            "--input",
            str(dev_csv),
            "--window",
            str(train_window),
            "--horizon",
            str(train_horizon),
            "--output-dir",
            str(model_out_dir),
            "--model",
            train_model,
            "--ridge-alpha",
            str(train_summary.get("ridge_alpha", 1.0)),
            "--stress-weight",
            str(train_summary.get("stress_weight", 2.0)),
            "--mixed-weight",
            str(train_summary.get("mixed_weight", 1.5)),
        ],
        cwd=root,
    )

    calibration_out = pred_results_dir / f"phase_calibration_{args.output_tag}.json"
    run_cmd(
        [
            sys.executable,
            str(calibration_script),
            "--input",
            str(holdout_csv),
            "--model-dir",
            str(model_out_dir),
            "--window",
            str(train_window),
            "--horizon",
            str(train_horizon),
            "--output",
            str(calibration_out),
        ],
        cwd=root,
    )

    # Dev-only evaluation using the locked final live policy family.
    selection_protocol = json.load(open(eval_protocol, "r", encoding="utf-8"))
    selection_protocol["data_split"] = {"freeze_test_runs": False, "test_runs": []}
    selection_protocol_path = results_dir / f"evaluation_protocol_selection_{args.output_tag}.json"
    with open(selection_protocol_path, "w", encoding="utf-8") as f:
        json.dump(selection_protocol, f, indent=2)

    selection_out = results_dir / f"offline_policy_selection_{args.output_tag}.json"
    run_cmd(
        [
            sys.executable,
            str(eval_script),
            "--runs-dir",
            str(runs_dir),
            "--model-dir",
            str(model_out_dir),
            "--protocol",
            str(selection_protocol_path),
            "--split-config",
            str(expanded_cfg_path),
            "--warmup",
            str(LOCKED_OFFLINE_WARMUP),
            "--anomaly-history",
            str(LOCKED_OFFLINE_ANOMALY_HISTORY),
            "--anomaly-source",
            LOCKED_OFFLINE_ANOMALY_SOURCE,
            "--nsa-num-detectors",
            str(LOCKED_OFFLINE_NSA_NUM_DETECTORS),
            "--nsa-radius",
            str(LOCKED_OFFLINE_NSA_RADIUS),
            "--weight-grid",
            str(LOCKED_OFFLINE_PRED_WEIGHT),
            "--z-grid",
            str(LOCKED_OFFLINE_Z_THRESHOLD),
            "--validation-runs",
            *dev_runs,
            "--test-runs",
            *dev_runs,
            "--output",
            str(selection_out),
        ],
        cwd=root,
    )

    selection_data = json.load(open(selection_out, "r", encoding="utf-8"))
    best = selection_data["best_validation_config"]
    selected_pred_weight = float(best["pred_weight"])
    selected_z = float(best["z_threshold"])
    print(
        f"Selection best config: pred_weight={selected_pred_weight}, "
        f"anomaly_weight={1.0-selected_pred_weight:.3f}, z={selected_z}"
    )

    # Fixed-parameter audit on strict holdout.
    sys.path.insert(0, str((root / "scheduler-prediction" / "custom-scheduler").resolve()))
    ope = load_offline_eval_module(eval_script)
    protocol = ope.load_protocol(str(eval_protocol))
    window_size = ope.resolve_window_size(str(model_out_dir), None)

    audit_result = ope.evaluate_split(
        runs_dir=str(runs_dir),
        run_ids=strict_holdout,
        model_dir=str(model_out_dir),
        window=window_size,
        warmup=LOCKED_OFFLINE_WARMUP,
        anomaly_history=LOCKED_OFFLINE_ANOMALY_HISTORY,
        z_threshold=selected_z,
        hybrid_pred_weight=selected_pred_weight,
        anomaly_source=LOCKED_OFFLINE_ANOMALY_SOURCE,
        nsa_num_detectors=LOCKED_OFFLINE_NSA_NUM_DETECTORS,
        nsa_radius=LOCKED_OFFLINE_NSA_RADIUS,
        protocol=protocol,
    )
    consistency_test = ope.evaluate_runwise_hybrid_consistency(
        runs_dir=str(runs_dir),
        run_ids=strict_holdout,
        model_dir=str(model_out_dir),
        window=window_size,
        warmup=LOCKED_OFFLINE_WARMUP,
        anomaly_history=LOCKED_OFFLINE_ANOMALY_HISTORY,
        z_threshold=selected_z,
        hybrid_pred_weight=selected_pred_weight,
        anomaly_source=LOCKED_OFFLINE_ANOMALY_SOURCE,
        nsa_num_detectors=LOCKED_OFFLINE_NSA_NUM_DETECTORS,
        nsa_radius=LOCKED_OFFLINE_NSA_RADIUS,
        protocol=protocol,
    )

    audit_out = results_dir / f"offline_policy_audit_{args.output_tag}.json"
    audit_payload = {
        "config": {
            "stage": "audit_fixed_config",
            "runs_dir": str(runs_dir),
            "model_dir": str(model_out_dir),
            "protocol_path": str(eval_protocol),
            "split_config_source": str(expanded_cfg_path),
            "selection_source": str(selection_out),
            "window": window_size,
            "warmup": LOCKED_OFFLINE_WARMUP,
            "anomaly_history": LOCKED_OFFLINE_ANOMALY_HISTORY,
            "anomaly_source": LOCKED_OFFLINE_ANOMALY_SOURCE,
            "nsa_num_detectors": LOCKED_OFFLINE_NSA_NUM_DETECTORS,
            "nsa_radius": LOCKED_OFFLINE_NSA_RADIUS,
            "validation_runs": dev_runs,
            "test_runs": strict_holdout,
        },
        "protocol": protocol,
        "best_validation_config": {
            "pred_weight": selected_pred_weight,
            "anomaly_weight": round(1.0 - selected_pred_weight, 6),
            "z_threshold": selected_z,
            "validation_summary": best.get("validation_summary"),
        },
        "consistency_test": consistency_test,
        "untouched_test_evaluation": audit_result,
    }
    with open(audit_out, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"Wrote fixed audit output: {audit_out}")

    # Compare against old baseline result.
    baseline_candidates = [
        results_dir / "offline_policy_evaluation_protocol_adaptive_weighted_calibrated_reconfirm.json",
        results_dir / "offline_policy_evaluation_protocol_adaptive_weighted_calibrated.json",
    ]
    baseline_path = next((p for p in baseline_candidates if p.exists()), None)
    if baseline_path is None:
        raise RuntimeError("Could not find baseline result file for old-vs-new comparison")

    old_data = json.load(open(baseline_path, "r", encoding="utf-8"))
    new_cmp = audit_result["comparisons"]["hybrid_vs_prediction_only"]

    old_best = old_data.get("best_validation_config", {})
    old_pred_weight = float(old_best.get("pred_weight", 0.9))
    old_z = float(old_best.get("z_threshold", 3.5))

    if fair_mode:
        old_cmp = ope.evaluate_split(
            runs_dir=str(runs_dir),
            run_ids=strict_holdout,
            model_dir=str(base_model_dir),
            window=ope.resolve_window_size(str(base_model_dir), None),
            warmup=LOCKED_OFFLINE_WARMUP,
            anomaly_history=LOCKED_OFFLINE_ANOMALY_HISTORY,
            z_threshold=old_z,
            hybrid_pred_weight=old_pred_weight,
            anomaly_source=LOCKED_OFFLINE_ANOMALY_SOURCE,
            nsa_num_detectors=LOCKED_OFFLINE_NSA_NUM_DETECTORS,
            nsa_radius=LOCKED_OFFLINE_NSA_RADIUS,
            protocol=protocol,
        )["comparisons"]["hybrid_vs_prediction_only"]
    else:
        old_cmp = old_data["untouched_test_evaluation"]["comparisons"]["hybrid_vs_prediction_only"]

    anomaly_present_runs = []
    for run_id in strict_holdout:
        lp = runs_dir / run_id / "labelled.csv"
        if not lp.exists():
            continue
        ldf = pd.read_csv(lp)
        if "label" not in ldf.columns:
            continue
        if int(ldf["label"].sum()) > 0:
            anomaly_present_runs.append(run_id)

    per_run = []
    for run_id in strict_holdout:
        old_r = ope.evaluate_split(
            runs_dir=str(runs_dir),
            run_ids=[run_id],
            model_dir=str(base_model_dir),
            window=ope.resolve_window_size(str(base_model_dir), None),
            warmup=LOCKED_OFFLINE_WARMUP,
            anomaly_history=LOCKED_OFFLINE_ANOMALY_HISTORY,
            z_threshold=old_z,
            hybrid_pred_weight=old_pred_weight,
            anomaly_source=LOCKED_OFFLINE_ANOMALY_SOURCE,
            nsa_num_detectors=LOCKED_OFFLINE_NSA_NUM_DETECTORS,
            nsa_radius=LOCKED_OFFLINE_NSA_RADIUS,
            protocol=protocol,
        )["comparisons"]["hybrid_vs_prediction_only"]

        new_r = ope.evaluate_split(
            runs_dir=str(runs_dir),
            run_ids=[run_id],
            model_dir=str(model_out_dir),
            window=window_size,
            warmup=LOCKED_OFFLINE_WARMUP,
            anomaly_history=LOCKED_OFFLINE_ANOMALY_HISTORY,
            z_threshold=selected_z,
            hybrid_pred_weight=selected_pred_weight,
            anomaly_source=LOCKED_OFFLINE_ANOMALY_SOURCE,
            nsa_num_detectors=LOCKED_OFFLINE_NSA_NUM_DETECTORS,
            nsa_radius=LOCKED_OFFLINE_NSA_RADIUS,
            protocol=protocol,
        )["comparisons"]["hybrid_vs_prediction_only"]

        per_run.append(
            {
                "run_id": run_id,
                "anomaly_present": run_id in anomaly_present_runs,
                "old_utility_delta": old_r["utility_delta"],
                "new_utility_delta": new_r["utility_delta"],
                "delta_new_minus_old": round(new_r["utility_delta"] - old_r["utility_delta"], 6),
                "old_safe_delta": old_r["safe_placement_rate_delta"],
                "new_safe_delta": new_r["safe_placement_rate_delta"],
                "old_anom_rate_delta": old_r["anomalous_rate_delta"],
                "new_anom_rate_delta": new_r["anomalous_rate_delta"],
            }
        )

    anom_slice = [r for r in per_run if r["anomaly_present"]]
    comparison = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_result": str(baseline_path),
        "new_result": str(audit_out),
        "baseline_best_config": {
            "pred_weight": old_pred_weight,
            "anomaly_weight": round(1.0 - old_pred_weight, 6),
            "z_threshold": old_z,
        },
        "new_best_config": {
            "pred_weight": selected_pred_weight,
            "anomaly_weight": round(1.0 - selected_pred_weight, 6),
            "z_threshold": selected_z,
        },
        "holdout_global": {
            "old": old_cmp,
            "new": new_cmp,
            "new_minus_old": {k: round(float(new_cmp[k]) - float(old_cmp[k]), 6) for k in new_cmp.keys()},
        },
        "anomaly_present_runs": anomaly_present_runs,
        "anomaly_slice_summary": {
            "count": len(anom_slice),
            "old_positive_utility_count": sum(1 for r in anom_slice if r["old_utility_delta"] > 0),
            "new_positive_utility_count": sum(1 for r in anom_slice if r["new_utility_delta"] > 0),
            "avg_old_utility_delta": round(sum(r["old_utility_delta"] for r in anom_slice) / len(anom_slice), 6)
            if anom_slice
            else None,
            "avg_new_utility_delta": round(sum(r["new_utility_delta"] for r in anom_slice) / len(anom_slice), 6)
            if anom_slice
            else None,
        },
        "per_run": per_run,
    }

    compare_out = results_dir / f"novel_expansion_comparison_{args.output_tag}.json"
    with open(compare_out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Pipeline complete ===")
    print(f"Novel plan: {plan_path}")
    print(f"Expanded split config: {expanded_cfg_path}")
    print(f"Expanded model dir: {model_out_dir}")
    print(f"Quality report: {quality_json}")
    print(f"Calibration report: {calibration_out}")
    print(f"Selection result: {selection_out}")
    print(f"Audit result: {audit_out}")
    print(f"Old-vs-new comparison: {compare_out}")


if __name__ == "__main__":
    main()
