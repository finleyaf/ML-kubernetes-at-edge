# Custom Scheduler (Anomaly + Prediction)

This module provides a lightweight hybrid node-ranking engine that combines:

- Predicted future load (from `prediction/predictor.py` models)
- Online anomaly risk (rolling z-score monitor per node)

## Files

- `hybrid_scheduler.py`: core scoring logic
- `simulate_scheduler.py`: replay-based evaluation on labelled telemetry
- `rank_live.py`: rank nodes from recent telemetry and return best target node

## Run simulation

From `project_folder/scheduler-prediction`:

```bash
python custom-scheduler/simulate_scheduler.py \
  --input ../anomaly-detection/online-telemetry/dataset/labelled.csv \
  --model-dir prediction/models \
  --window 10 \
  --warmup 15 \
  --pred-weight 0.9 \
  --anomaly-weight 0.1 \
  --anomaly-history 45 \
  --z-threshold 3.5 \
  --output custom-scheduler/results/simulation_summary.json
```

## Rank nodes from recent telemetry

```bash
python custom-scheduler/rank_live.py \
  --input ../anomaly-detection/online-telemetry/dataset/dataset.csv \
  --model-dir prediction/models \
  --window 10 \
  --pred-weight 0.9 \
  --anomaly-weight 0.1 \
  --anomaly-history 45 \
  --z-threshold 3.5 \
  --output custom-scheduler/results/latest_ranking.json
```

## Output

The output JSON includes:

- `hybrid_scheduler` summary metrics
- `prediction_only_scheduler` baseline metrics
- `comparison` deltas (safe-placement rate and anomalous placements)
