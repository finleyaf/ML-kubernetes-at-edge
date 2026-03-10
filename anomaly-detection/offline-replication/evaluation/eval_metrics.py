import pandas as pd
from sklearn.metrics import precision_score, recall_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)

y_true = df["label"]
y_pred = df["nsa_anomaly"]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))