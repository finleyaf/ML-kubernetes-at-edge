#!/usr/bin/env python3
import argparse
import json
import subprocess


def parse_cpu(raw: str) -> float:
    value = str(raw).strip()
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000.0


def parse_memory(raw: str) -> float:
    value = str(raw).strip()
    units = {
        "Ki": 1.0 / 1024.0,
        "Mi": 1.0,
        "Gi": 1024.0,
        "Ti": 1024.0 * 1024.0,
    }
    for suffix, scale in units.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * scale
    return float(value)


def fetch_nodes(control_node: str, zone: str) -> dict:
    command = [
        "gcloud",
        "compute",
        "ssh",
        control_node,
        f"--zone={zone}",
        '--command=kubectl get nodes -o json',
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export allocatable node capacity from the cluster")
    parser.add_argument("--control-node", required=True, help="Control-plane VM name used for kubectl access")
    parser.add_argument("--zone", required=True, help="GCE zone for the control-plane VM")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--nodes", nargs="*", help="Optional subset of node names to export")
    args = parser.parse_args()

    payload = fetch_nodes(args.control_node, args.zone)
    selected = set(args.nodes or [])
    capacities = {}

    for item in payload.get("items", []):
        name = item.get("metadata", {}).get("name")
        if not name:
            continue
        if selected and name not in selected:
            continue

        allocatable = item.get("status", {}).get("allocatable", {})
        capacities[name] = {
            "cpu_millicores": round(parse_cpu(allocatable.get("cpu", "0")), 4),
            "memory_mib": round(parse_memory(allocatable.get("memory", "0")), 4),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(capacities, f, indent=2)

    print(f"saved {args.output}")
    print(f"nodes {len(capacities)}")


if __name__ == "__main__":
    main()