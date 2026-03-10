import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data/processed.csv")

features = ["cpu", "mem", "net"]
df["kmeans_anomaly"] = 0

for f in features:
    km = KMeans(n_clusters=2, random_state=0)
    labels = km.fit_predict(df[[f]])
    normal_cluster = pd.Series(labels).value_counts().idxmax()
    df[f"{f}_kmeans_anomaly"] = (labels != normal_cluster).astype(int)

anomaly_cols = [f"{f}_kmeans_anomaly" for f in features]
# Use NumPy array max to avoid pandas axis typing ambiguity
df["kmeans_anomaly"] = df[anomaly_cols].to_numpy().max(axis=1)

df.to_csv("data/kmeans_output.csv", index=False)
print("K-means detection results saved to data/kmeans_output.csv")