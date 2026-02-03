import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--window", type=int, default=10)
args = parser.parse_args()

df = pd.read_csv(args.input)
# rolling mean smoothing
features = ["cpu", "mem", "net"]
for feature in features:
    df[feature] = df[feature].rolling(window=args.window, min_periods=1).mean()

df.to_csv(args.output, index=False)
print(f"Processed data saved to {args.output} with window size {args.window}")