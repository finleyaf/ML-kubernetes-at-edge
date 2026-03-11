import pandas as pd
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to scheduling results CSV")
parser.add_argument("--output", required=True, help="Path to save analysis JSON")
args = parser.parse_args()

df = pd.read_csv(args.input)

# split by condition (normal vs stress based on pod name)
df["condition"] = df["pod_name"].apply(
    lambda x: "stress" if "stress" in x else "normal"
)

results = {}

for condition in df["condition"].unique():
    cond_df = df[df["condition"] == condition]

    # node placement distribution
    node_counts = cond_df["node"].value_counts().to_dict()

    # timing metrics per workload type
    workload_stats = {}
    for wtype in cond_df["workload_type"].unique():
        wdf = cond_df[cond_df["workload_type"] == wtype]
        workload_stats[wtype] = {
            "count": len(wdf),
            "scheduling_latency": {
                "mean": round(wdf["scheduling_latency_s"].mean(), 4),
                "std": round(wdf["scheduling_latency_s"].std(), 4),
                "min": round(wdf["scheduling_latency_s"].min(), 4),
                "max": round(wdf["scheduling_latency_s"].max(), 4)
            },
            "startup_time": {
                "mean": round(wdf["startup_time_s"].mean(), 4),
                "std": round(wdf["startup_time_s"].std(), 4),
                "min": round(wdf["startup_time_s"].min(), 4),
                "max": round(wdf["startup_time_s"].max(), 4)
            },
            "total_time": {
                "mean": round(wdf["total_time_s"].mean(), 4),
                "std": round(wdf["total_time_s"].std(), 4),
                "min": round(wdf["total_time_s"].min(), 4),
                "max": round(wdf["total_time_s"].max(), 4)
            }
        }

    # overall stats
    results[condition] = {
        "total_pods": len(cond_df),
        "node_placements": node_counts,
        "avg_scheduling_latency_s": round(cond_df["scheduling_latency_s"].mean(), 4),
        "avg_startup_time_s": round(cond_df["startup_time_s"].mean(), 4),
        "avg_total_time_s": round(cond_df["total_time_s"].mean(), 4),
        "workloads": workload_stats
    }

# summary comparison
if "normal" in results and "stress" in results:
    results["comparison"] = {
        "scheduling_latency_increase": round(
            results["stress"]["avg_scheduling_latency_s"] - results["normal"]["avg_scheduling_latency_s"], 4
        ),
        "startup_time_increase": round(
            results["stress"]["avg_startup_time_s"] - results["normal"]["avg_startup_time_s"], 4
        ),
        "stress_node_placement_pct": round(
            results["stress"]["node_placements"].get("k3s-worker-2", 0) / results["stress"]["total_pods"] * 100, 1
        )
    }

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)

# print summary
print("\n=== Baseline Scheduler Analysis ===\n")
for condition, data in results.items():
    if condition == "comparison":
        continue
    print(f"Condition: {condition}")
    print(f"  Pods scheduled:        {data['total_pods']}")
    print(f"  Node placements:       {data['node_placements']}")
    print(f"  Avg scheduling latency: {data['avg_scheduling_latency_s']}s")
    print(f"  Avg startup time:       {data['avg_startup_time_s']}s")
    print(f"  Avg total time:         {data['avg_total_time_s']}s")
    print()

if "comparison" in results:
    comp = results["comparison"]
    print("Comparison (stress vs normal):")
    print(f"  Scheduling latency increase: {comp['scheduling_latency_increase']}s")
    print(f"  Startup time increase:       {comp['startup_time_increase']}s")
    print(f"  % pods placed on stressed node: {comp['stress_node_placement_pct']}%")

print(f"\nFull results saved to {args.output}")
