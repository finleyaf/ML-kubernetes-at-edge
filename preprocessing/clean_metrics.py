import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to raw dataset CSV")
parser.add_argument("--output", required=True, help="Path to save processed CSV")
parser.add_argument("--window", type=int, default=10, help="Rolling mean window size")
parser.add_argument("--node", type=str, default=None, help="Filter to a specific node (e.g. k3s-worker-2)")
args = parser.parse_args()

df = pd.read_csv(args.input)

# optional node filter
if args.node:
    df = df[df["node"] == args.node].copy()
    print(f"Filtered to node: {args.node} ({len(df)} rows)")

# combine cpu_user + cpu_system into a single cpu feature (matching synthetic 'cpu')
df["cpu"] = df["cpu_user"] + df["cpu_system"]

# use ram_used as mem feature (matching synthetic 'mem')
df["mem"] = df["ram_used"]

# combine net_received + net_sent into a single net feature (matching synthetic 'net')
df["net"] = df["net_received"] + df["net_sent"]

features = ["cpu", "mem", "net"]

# process each node separately to avoid cross-node smoothing
processed_frames = []
for node in df["node"].unique():
    node_df = df[df["node"] == node].copy().sort_values("timestamp")

    # rolling mean smoothing
    for feature in features:
        node_df[feature] = node_df[feature].rolling(window=args.window, min_periods=1).mean()

    # min-max normalisation per feature
    for feature in features:
        fmin = node_df[feature].min()
        fmax = node_df[feature].max()
        if fmax - fmin > 0:
            node_df[feature] = (node_df[feature] - fmin) / (fmax - fmin)
        else:
            node_df[feature] = 0.0

    processed_frames.append(node_df)

result = pd.concat(processed_frames).sort_values("timestamp")

# keep only the columns needed for detection
output_cols = ["timestamp", "node", "cpu", "mem", "net"]
if "label" in result.columns:
    output_cols.append("label")

result[output_cols].to_csv(args.output, index=False)
print(f"Processed data saved to {args.output} (window={args.window})")
