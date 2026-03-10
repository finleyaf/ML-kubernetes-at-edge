import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to dataset CSV")
parser.add_argument("--output", required=True, help="Path to save labelled CSV")
parser.add_argument("--phases", required=True, help="Path to phases JSON file")
args = parser.parse_args()

df = pd.read_csv(args.input)

with open(args.phases) as f:
    phases = json.load(f)

# initialise all labels as normal (0)
df["label"] = 0

# mark stress phases as anomalous (1)
for phase in phases:
    if phase["type"] == "stress":
        mask = (df["timestamp"] >= phase["start"]) & (df["timestamp"] <= phase["end"])
        df.loc[mask, "label"] = 1

normal_count = (df["label"] == 0).sum()
anomaly_count = (df["label"] == 1).sum()
print(f"Labelled {len(df)} rows: {normal_count} normal, {anomaly_count} anomalous")

df.to_csv(args.output, index=False)
print(f"Labelled data saved to {args.output}")
