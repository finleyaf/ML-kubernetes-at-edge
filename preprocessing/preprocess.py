import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=10)
args = parser.parse_args()

df = pd.read_csv("data/metrics.csv")

# rolling mean smoothing
features = ["cpu", "mem", "net"]
for feature in features:
    df[feature] = df[feature].rolling(window=args.window, min_periods=1).mean()

df.to_csv("data/processed.csv", index=False)
print(f"Processed data saved to data/processed.csv with window size {args.window}")