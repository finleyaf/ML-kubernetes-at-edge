import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


DEFAULT_STRESS_POD_IMAGE = "alpine:3.20"


def run_shell(command, cwd=None):
    subprocess.run(command, shell=True, check=True, cwd=cwd)


def run_gcloud_ssh(node, zone, command, retries=3, retry_delay=15, capture_output=False, input_text=None):
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        node,
        f"--zone={zone}",
        f"--command={command}",
    ]
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                input=input_text,
                capture_output=capture_output,
            )
            if capture_output:
                return result.stdout.strip()
            return None
        except subprocess.CalledProcessError as e:
            last_error = e
            if attempt >= retries:
                break
            print(
                f"Remote command failed on {node} (attempt {attempt}/{retries}), "
                f"retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)

    if last_error is None:
        raise RuntimeError(f"Remote command failed on {node} without a captured exception")
    raise last_error


def quote_remote_args(args):
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run_remote_kubectl(control_node, zone, kubectl_args, retries=3, retry_delay=15, capture_output=False, input_text=None):
    return run_gcloud_ssh(
        control_node,
        zone,
        f"kubectl {quote_remote_args(kubectl_args)}",
        retries=retries,
        retry_delay=retry_delay,
        capture_output=capture_output,
        input_text=input_text,
    )


def normalise_stress_command(command):
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Empty stress command")
    if tokens[0] == "stress":
        tokens[0] = "stress-ng"
    for idx, token in enumerate(tokens[:-1]):
        if token == "--timeout":
            timeout_value = tokens[idx + 1]
            if timeout_value.isdigit():
                tokens[idx + 1] = f"{timeout_value}s"
    return shlex.join(tokens)


def make_stress_pod_name(run_id, phase_name, target):
    raw_name = f"stress-{run_id}-{phase_name}-{target}".lower()
    clean = re.sub(r"[^a-z0-9-]", "-", raw_name)
    clean = re.sub(r"-+", "-", clean).strip("-")
    digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:6]
    prefix = clean[: max(1, 63 - len(digest) - 1)].rstrip("-")
    return f"{prefix}-{digest}"


def build_stress_pod_manifest(pod_name, target, image, stress_command):
    return f"""apiVersion: v1
kind: Pod
metadata:
  name: {pod_name}
spec:
  restartPolicy: Never
  nodeName: {target}
  tolerations:
  - operator: Exists
  containers:
  - name: stress
    image: {image}
    command: [\"/bin/sh\", \"-lc\"]
    args:
    - |
      set -eu
      apk add --no-cache stress-ng >/dev/null
      exec {stress_command}
"""


def wait_for_stress_pod(control_node, zone, pod_name, duration, ready_timeout, completion_buffer, retries, retry_delay):
    deadline = time.time() + max(ready_timeout, duration + completion_buffer)
    ready_seen = False

    while time.time() < deadline:
        phase = run_remote_kubectl(
            control_node,
            zone,
            ["get", "pod", pod_name, "-o", "jsonpath={.status.phase}"],
            retries=retries,
            retry_delay=retry_delay,
            capture_output=True,
        )
        phase = (phase or "").strip()
        if phase in {"Running", "Succeeded"}:
            ready_seen = True
        if phase == "Succeeded":
            return
        if phase == "Failed":
            logs = run_remote_kubectl(
                control_node,
                zone,
                ["logs", pod_name],
                retries=1,
                retry_delay=0,
                capture_output=True,
            )
            raise RuntimeError(f"stress pod {pod_name} failed:\n{logs}")
        if not ready_seen and time.time() >= deadline:
            break
        time.sleep(2)

    raise RuntimeError(f"stress pod {pod_name} did not complete within timeout")


def run_stress_via_pod(target, zone, control_node, image, command, duration, retries, retry_delay, ready_timeout, completion_buffer):
    pod_name = make_stress_pod_name("run", str(int(time.time())), target)
    manifest = build_stress_pod_manifest(pod_name, target, image, normalise_stress_command(command))

    try:
        run_remote_kubectl(
            control_node,
            zone,
            ["apply", "-f", "-"],
            retries=retries,
            retry_delay=retry_delay,
            input_text=manifest,
        )
        wait_for_stress_pod(
            control_node,
            zone,
            pod_name,
            duration,
            ready_timeout,
            completion_buffer,
            retries,
            retry_delay,
        )
    finally:
        try:
            run_remote_kubectl(
                control_node,
                zone,
                ["delete", "pod", pod_name, "--ignore-not-found=true"],
                retries=1,
                retry_delay=0,
            )
        except Exception:
            pass


def run_stress(target, args, command, duration):
    if args.stress_executor == "ssh":
        run_gcloud_ssh(
            target,
            args.zone,
            command,
            retries=args.ssh_retries,
            retry_delay=args.ssh_retry_delay,
        )
        return

    run_stress_via_pod(
        target=target,
        zone=args.zone,
        control_node=args.control_node,
        image=args.stress_pod_image,
        command=command,
        duration=duration,
        retries=args.ssh_retries,
        retry_delay=args.ssh_retry_delay,
        ready_timeout=args.stress_ready_timeout,
        completion_buffer=args.stress_completion_buffer,
    )


def load_plan(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def file_has_data_rows(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            # header + at least one data row
            return sum(1 for _ in f) > 1
    except Exception:
        return False


def collector_startup_check(collector, raw_csv, timeout_s, log_path):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if collector.poll() is not None:
            msg = ""
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    msg = "\n".join(f.readlines()[-20:])
            raise RuntimeError(
                "Collector process exited during startup. "
                f"Check log: {log_path}\n{msg}"
            )

        if file_has_data_rows(raw_csv):
            return

        time.sleep(1)

    raise RuntimeError(
        "Collector did not produce telemetry rows during startup window. "
        f"Check log: {log_path}"
    )


def execute_run(run, args, project_dir):
    run_id = run["run_id"]
    run_dir = os.path.join(project_dir, "dataset", "runs", run_id)
    ensure_dir(run_dir)

    raw_csv = os.path.join(run_dir, "dataset.csv")
    labelled_csv = os.path.join(run_dir, "labelled.csv")
    phases_json = os.path.join(run_dir, "phases.json")
    metadata_json = os.path.join(run_dir, "metadata.json")
    collector_log = os.path.join(run_dir, "collector.log")

    collector_cmd = [
        sys.executable,
        os.path.join(project_dir, "data_collection", "netdata_collector.py"),
        "--base-url",
        args.base_url,
        "--output",
        raw_csv,
        "--interval",
        str(args.interval),
        "--nodes",
        *args.nodes,
    ]

    print(f"\n=== Starting run {run_id} ({run['kind']}) ===")
    print(f"Output directory: {run_dir}")

    log_handle = open(collector_log, "w")
    collector = subprocess.Popen(
        collector_cmd,
        stdout=log_handle,
        stderr=log_handle,
        preexec_fn=os.setsid,
    )
    collector_startup_check(
        collector=collector,
        raw_csv=raw_csv,
        timeout_s=args.collector_startup_timeout,
        log_path=collector_log,
    )

    phase_records = []
    try:
        for phase in run["phases"]:
            phase_type = phase["type"]
            phase_name = phase["name"]
            duration = int(phase["duration"])

            start = int(time.time())
            print(f"[{run_id}] Phase: {phase_name} ({phase_type}), duration={duration}s")

            if phase_type == "stress":
                run_stress(phase["target"], args, phase["command"], duration)
            else:
                time.sleep(duration)

            end = int(time.time())

            record = {
                "type": phase_type,
                "name": phase_name,
                "start": start,
                "end": end,
            }
            if phase_type == "stress":
                record["target"] = phase["target"]
                record["intensity"] = phase["intensity"]
            phase_records.append(record)

    finally:
        print(f"[{run_id}] Stopping collector")
        try:
            os.killpg(os.getpgid(collector.pid), signal.SIGTERM)
        except Exception:
            pass
        collector.wait(timeout=15)
        log_handle.close()
        time.sleep(1)

    if not file_has_data_rows(raw_csv):
        raise RuntimeError(
            f"Run {run_id} has no telemetry rows in {raw_csv}. "
            f"See collector log: {collector_log}"
        )

    save_json(phases_json, phase_records)

    label_cmd = [
        sys.executable,
        os.path.join(project_dir, "preprocessing", "label_data.py"),
        "--input",
        raw_csv,
        "--output",
        labelled_csv,
        "--phases",
        phases_json,
    ]
    subprocess.run(label_cmd, check=True)

    metadata = {
        "run_id": run_id,
        "kind": run["kind"],
        "order": run.get("order"),
        "zone": args.zone,
        "nodes": args.nodes,
        "collector_interval_s": args.interval,
        "stress_executor": args.stress_executor,
        "control_node": args.control_node,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "raw": raw_csv,
            "labelled": labelled_csv,
            "phases": phases_json,
            "collector_log": collector_log,
        },
    }
    save_json(metadata_json, metadata)

    print(f"[{run_id}] Completed")
    return {
        "run_id": run_id,
        "kind": run["kind"],
        "run_dir": run_dir,
        "raw_csv": raw_csv,
        "labelled_csv": labelled_csv,
        "phases_json": phases_json,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Campaign plan JSON")
    parser.add_argument("--zone", default="europe-west2-c", help="GCP zone")
    parser.add_argument(
        "--nodes",
        nargs="+",
        default=[
            "k3s-control",
            "k3s-worker-2",
            "k3s-worker-3",
            "k3s-worker-4",
            "raspberrypi",
        ],
        help="Nodes for metric collection",
    )
    parser.add_argument("--base-url", default="http://localhost:20000", help="Netdata base URL for the collector")
    parser.add_argument("--interval", type=float, default=1.0, help="Collector polling interval")
    parser.add_argument("--ssh-retries", type=int, default=3, help="Retries for transient gcloud ssh failures")
    parser.add_argument("--ssh-retry-delay", type=int, default=15, help="Seconds between ssh retry attempts")
    parser.add_argument(
        "--stress-executor",
        choices=["ssh", "k8s-pod"],
        default="ssh",
        help="How to execute stress commands for stress phases",
    )
    parser.add_argument("--control-node", default="k3s-control", help="Node to run kubectl from when using k8s-pod stress")
    parser.add_argument("--stress-pod-image", default=DEFAULT_STRESS_POD_IMAGE, help="Container image for k8s-pod stress")
    parser.add_argument("--stress-ready-timeout", type=int, default=60, help="Seconds to wait for a stress pod to start")
    parser.add_argument(
        "--stress-completion-buffer",
        type=int,
        default=90,
        help="Additional seconds beyond the planned stress duration before a stress pod is treated as stuck",
    )
    parser.add_argument(
        "--collector-startup-timeout",
        type=int,
        default=12,
        help="Seconds to wait for first telemetry rows before aborting a run",
    )
    parser.add_argument("--limit", type=int, help="Execute only first N runs from plan")
    parser.add_argument("--start-at", type=int, default=1, help="Start from run order index (1-based)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    plan = load_plan(args.plan)
    runs = sorted(plan["runs"], key=lambda r: r.get("order", 0))

    if args.start_at > 1:
        runs = runs[args.start_at - 1:]
    if args.limit is not None:
        runs = runs[: args.limit]

    manifest = {
        "plan": os.path.abspath(args.plan),
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "zone": args.zone,
        "nodes": args.nodes,
        "base_url": args.base_url,
        "interval": args.interval,
        "stress_executor": args.stress_executor,
        "runs": [],
    }

    for run in runs:
        rec = execute_run(run, args, project_dir)
        manifest["runs"].append(rec)

    manifest_path = os.path.join(project_dir, "dataset", "runs", "manifest.json")
    ensure_dir(os.path.dirname(manifest_path))
    save_json(manifest_path, manifest)
    print(f"\nCampaign complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
