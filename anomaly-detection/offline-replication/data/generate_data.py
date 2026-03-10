import numpy as np
import pandas as pd
import argparse

np.random.seed(42)

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["subtle", "sustained"],
    required=True,
    help="Type of anomaly injection"
)
args = parser.parse_args()

# ensure output filename is always bound
outfile = f"data/metrics_{args.mode}.csv"

# time axis
n_points = 2000
time = np.arange(n_points)

# normal behaviour
cpu = np.random.normal(30, 5, n_points)
mem = np.random.normal(40, 7, n_points)
net = np.random.normal(100, 15, n_points)

labels = np.zeros(n_points)

# inject anomalies
anomaly_idx = np.random.choice(n_points - 30, size=20, replace=False)

if args.mode == "subtle":
    for i in anomaly_idx:
        cpu[i] += np.random.normal(40, 5)
        mem[i] += np.random.normal(-20, 5)
        net[i] += np.random.normal(120, 20)
        #cpu[anomaly_idx] += 40
        #mem[anomaly_idx] -= 25
        #net[anomaly_idx] += 150
        labels[i] = 1

elif args.mode == "sustained":
    for i in anomaly_idx:
        cpu[i:i+30] += 50
        mem[i:i+30] -= 30
        net[i:i+30] += 200
        labels[i:i+30] = 1

df = pd.DataFrame({
    "time": time,
    "cpu": cpu,
    "mem": mem,
    "net": net,
    "label": labels
})

df.to_csv(outfile, index=False)
print(f"Generated dataset: {outfile}")