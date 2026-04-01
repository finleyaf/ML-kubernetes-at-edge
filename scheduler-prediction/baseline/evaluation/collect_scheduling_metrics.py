import subprocess
import json
import csv
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pod", required=True, help="Pod name to monitor")
parser.add_argument("--output", required=True, help="Path to append results CSV")
parser.add_argument("--timeout", type=int, default=120, help="Max seconds to wait for pod completion")
parser.add_argument("--control-node", required=True, help="GCE control node name (e.g. k3s-control)")
parser.add_argument("--zone", required=True, help="GCE zone (e.g. europe-west2-c)")
parser.add_argument(
    "--contention-relative-cpu-threshold",
    type=float,
    default=1.1,
    help="Relative CPU threshold above which a decision is marked high contention",
)
parser.add_argument(
    "--anomaly-relative-cpu-threshold",
    type=float,
    default=1.25,
    help="Relative CPU threshold above which a placement is marked anomalous",
)
args = parser.parse_args()


def kubectl_json(kubectl_cmd):
    """Run a kubectl command on the remote control node and return parsed JSON."""
    remote_cmd = f'gcloud compute ssh {args.control_node} --zone={args.zone} --command="{kubectl_cmd}"'
    result = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kubectl error: {result.stderr.strip()}")
    return json.loads(result.stdout)


def get_pod_events(pod_name):
    """Get scheduling-related events for a pod."""
    data = kubectl_json(
        f"kubectl get events --field-selector involvedObject.name={pod_name} -o json"
    )
    return data.get("items", [])


def parse_cpu_cores(raw):
    s = str(raw).strip().lower()
    if not s:
        return None
    if s.endswith("m"):
        return float(s[:-1]) / 1000.0
    return float(s)


def parse_cpu_percent(raw):
    s = str(raw).strip().lower()
    if not s:
        return None
    if s.endswith("%"):
        return float(s[:-1])
    return float(s)


def get_node_cpu_snapshot():
    """Return node CPU usage snapshot from metrics-server via kubectl top nodes."""
    remote_cmd = (
        f'gcloud compute ssh {args.control_node} --zone={args.zone} '
        '--command="kubectl top nodes --no-headers"'
    )
    result = subprocess.run(remote_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        return {}

    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        node = parts[0]
        rows[node] = {
            "cpu_cores": parse_cpu_cores(parts[1]),
            "cpu_percent": parse_cpu_percent(parts[2]),
        }
    return rows


def parse_timestamp(ts):
    """Parse k8s timestamp to epoch seconds."""
    from datetime import datetime
    # handle both formats: with and without microseconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def collect_pod_metrics(pod_name, timeout):
    """Wait for pod to schedule and run, then collect timing metrics."""

    # wait for pod to be scheduled
    print(f"Waiting for {pod_name} to be scheduled...")
    scheduled_time = None
    started_time = None
    node = None

    for _ in range(timeout):
        try:
            pod = kubectl_json(f"kubectl get pod {pod_name} -o json")
        except RuntimeError:
            time.sleep(1)
            continue

        phase = pod.get("status", {}).get("phase", "")
        node = pod.get("spec", {}).get("nodeName")

        # get condition timestamps
        conditions = pod.get("status", {}).get("conditions", [])
        for cond in conditions:
            if cond.get("type") == "PodScheduled" and cond.get("status") == "True":
                scheduled_time = parse_timestamp(cond["lastTransitionTime"])
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                started_time = parse_timestamp(cond["lastTransitionTime"])

        # get container start time
        container_statuses = pod.get("status", {}).get("containerStatuses", [])
        for cs in container_statuses:
            state = cs.get("state", {})
            running = state.get("running", {})
            if running.get("startedAt"):
                started_time = parse_timestamp(running["startedAt"])

        if phase in ("Succeeded", "Failed") or (scheduled_time and started_time):
            break

        time.sleep(1)

    # get creation timestamp
    creation_time = parse_timestamp(
        pod.get("metadata", {}).get("creationTimestamp", "")
    )

    # calculate durations
    scheduling_latency = None
    startup_time = None
    total_time = None

    if creation_time and scheduled_time:
        scheduling_latency = round(scheduled_time - creation_time, 3)
    if scheduled_time and started_time:
        startup_time = round(started_time - scheduled_time, 3)
    if creation_time and started_time:
        total_time = round(started_time - creation_time, 3)

    # get workload type from labels
    labels = pod.get("metadata", {}).get("labels", {})
    workload_type = labels.get("workload-type", "unknown")

    snapshot = get_node_cpu_snapshot()
    workers = [
        n for n in snapshot.keys()
        if "control" not in str(n).lower() and "master" not in str(n).lower()
    ]
    worker_cpu_pct = [snapshot[n]["cpu_percent"] for n in workers if snapshot[n]["cpu_percent"] is not None]

    chosen_observed_cpu = None
    chosen_relative_cpu = None
    high_contention = None
    chosen_label = None

    if node and node in snapshot:
        chosen_observed_cpu = snapshot[node]["cpu_percent"]
    if chosen_observed_cpu is not None and worker_cpu_pct:
        avg_cpu = sum(worker_cpu_pct) / max(1, len(worker_cpu_pct))
        if avg_cpu > 0:
            chosen_relative_cpu = round(chosen_observed_cpu / avg_cpu, 4)
            high_contention = 1 if chosen_relative_cpu >= args.contention_relative_cpu_threshold else 0
            chosen_label = 1 if chosen_relative_cpu >= args.anomaly_relative_cpu_threshold else 0

    return {
        "pod_name": pod_name,
        "workload_type": workload_type,
        "node": node,
        "scheduling_latency_s": scheduling_latency,
        "startup_time_s": startup_time,
        "total_time_s": total_time,
        "creation_time": creation_time,
        "scheduled_time": scheduled_time,
        "started_time": started_time,
        "chosen_observed_cpu": chosen_observed_cpu,
        "chosen_relative_cpu": chosen_relative_cpu,
        "high_contention": high_contention,
        "chosen_label": chosen_label,
    }


# collect metrics
metrics = collect_pod_metrics(args.pod, args.timeout)

# write to CSV
os.makedirs(os.path.dirname(args.output), exist_ok=True)
file_exists = os.path.exists(args.output)

with open(args.output, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=metrics.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(metrics)

print(f"Pod: {metrics['pod_name']}")
print(f"  Node:               {metrics['node']}")
print(f"  Workload type:      {metrics['workload_type']}")
print(f"  Scheduling latency: {metrics['scheduling_latency_s']}s")
print(f"  Startup time:       {metrics['startup_time_s']}s")
print(f"  Total time:         {metrics['total_time_s']}s")
