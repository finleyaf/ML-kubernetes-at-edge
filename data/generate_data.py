import numpy as np
import pandas as pd

np.random.seed(42)

# time axis
n_points = 2000
time = np.arange(n_points)

# normal behaviour
cpu = np.random.normal(30, 5, n_points)
mem = np.random.normal(40, 7, n_points)
net = np.random.normal(100, 15, n_points)

# inject anomalies
anomaly_idx = np.random.choice(n_points, size=60, replace=False)
cpu[anomaly_idx] += np.random.normal(50, 10, len(anomaly_idx))
mem[anomaly_idx] += np.random.normal(30, 5, len(anomaly_idx))
net[anomaly_idx] += np.random.normal(200, 30, len(anomaly_idx))

labels = np.zeros(n_points)
labels[anomaly_idx] = 1

df = pd.DataFrame({
    "time": time,
    "cpu": cpu,
    "mem": mem,
    "net": net,
    "label": labels
})

df.to_csv("data/metrics.csv", index=False)
print("Synthetic data saved to data/metrics.csv")