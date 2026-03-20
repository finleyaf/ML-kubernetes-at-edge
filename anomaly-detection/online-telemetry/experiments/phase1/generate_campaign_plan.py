import argparse
import json
import os
import random
from datetime import datetime, timezone

STRESS_LIBRARY = [
    {
        "name": "cpu",
        "target": "k3s-worker-2",
        "profiles": [
            {"id": "low", "command": "stress --cpu 1 --timeout {duration}"},
            {"id": "medium", "command": "stress --cpu 2 --timeout {duration}"},
            {"id": "high", "command": "stress --cpu 3 --timeout {duration}"},
        ],
    },
    {
        "name": "memory",
        "target": "k3s-control",
        "profiles": [
            {"id": "low", "command": "stress --vm 1 --vm-bytes 512M --timeout {duration}"},
            {"id": "medium", "command": "stress --vm 1 --vm-bytes 1G --timeout {duration}"},
            {"id": "high", "command": "stress --vm 2 --vm-bytes 1G --timeout {duration}"},
        ],
    },
    {
        "name": "io",
        "target": "k3s-worker-2",
        "profiles": [
            {"id": "low", "command": "stress --io 2 --timeout {duration}"},
            {"id": "medium", "command": "stress --io 4 --timeout {duration}"},
            {"id": "high", "command": "stress --io 6 --timeout {duration}"},
        ],
    },
    {
        "name": "mixed",
        "target": "k3s-worker-3",
        "profiles": [
            {"id": "low", "command": "stress --cpu 1 --vm 1 --vm-bytes 256M --timeout {duration}"},
            {"id": "medium", "command": "stress --cpu 1 --vm 1 --vm-bytes 512M --timeout {duration}"},
            {"id": "high", "command": "stress --cpu 2 --vm 1 --vm-bytes 1G --timeout {duration}"},
        ],
    },
]


def build_run(run_idx, is_control, baseline_s, recovery_s, duration_options, rng):
    run_id = f"run_{run_idx:03d}"
    phases = [
        {"type": "baseline", "name": "baseline", "duration": baseline_s},
    ]

    if not is_control:
        events = STRESS_LIBRARY.copy()
        rng.shuffle(events)

        for event in events:
            duration = rng.choice(duration_options)
            profile = rng.choice(event["profiles"])
            phases.append(
                {
                    "type": "stress",
                    "name": event["name"],
                    "target": event["target"],
                    "intensity": profile["id"],
                    "duration": duration,
                    "command": profile["command"].format(duration=duration),
                }
            )

    phases.append({"type": "recovery", "name": "recovery", "duration": recovery_s})

    return {
        "run_id": run_id,
        "kind": "control" if is_control else "stress",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phases": phases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20, help="Total number of runs")
    parser.add_argument("--control-ratio", type=float, default=0.25, help="Fraction of no-stress control runs")
    parser.add_argument("--baseline", type=int, default=120, help="Baseline duration in seconds")
    parser.add_argument("--recovery", type=int, default=120, help="Recovery duration in seconds")
    parser.add_argument("--durations", nargs="+", type=int, default=[90, 120, 150], help="Stress durations to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="Path to save campaign plan JSON")
    args = parser.parse_args()

    if not (0.0 <= args.control_ratio <= 1.0):
        raise ValueError("--control-ratio must be between 0 and 1")

    rng = random.Random(args.seed)
    run_count = args.runs
    control_count = int(round(run_count * args.control_ratio))
    stress_count = run_count - control_count

    runs = []
    for i in range(1, stress_count + 1):
        runs.append(build_run(i, False, args.baseline, args.recovery, args.durations, rng))
    for i in range(stress_count + 1, run_count + 1):
        runs.append(build_run(i, True, args.baseline, args.recovery, args.durations, rng))

    rng.shuffle(runs)

    for idx, run in enumerate(runs, start=1):
        run["order"] = idx

    output = {
        "meta": {
            "runs": run_count,
            "control_ratio": args.control_ratio,
            "seed": args.seed,
            "baseline": args.baseline,
            "recovery": args.recovery,
            "durations": args.durations,
        },
        "runs": runs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved campaign plan: {args.output}")
    print(f"Stress runs: {stress_count}, Control runs: {control_count}")


if __name__ == "__main__":
    main()
