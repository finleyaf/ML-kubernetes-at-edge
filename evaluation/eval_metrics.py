import pandas as pd
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("data/nsa_output.csv")

y_true = df["label"]
y_pred = df["nsa_anomaly"]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))