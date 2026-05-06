# Custom Scheduler (Anomaly + Prediction)

This module provides a lightweight hybrid node-ranking engine that combines:

- Predicted future load (from `prediction/predictor.py` models)
- Online anomaly risk from z-score, NSA, or k-means detectors

## Files

- `hybrid_scheduler.py`: core scoring logic
- `simulate_scheduler.py`: replay-based evaluation on labelled telemetry
- `rank_live.py`: rank nodes from recent telemetry and return best target node
- `offline_policy_evaluation.py`: paired offline comparison for prediction-only, anomaly-only, and hybrid policies, with defaults aligned to the locked final live policy family

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
  --anomaly-source nsa \
  --pred-weight 0.9 \
  --anomaly-weight 0.1 \
  --anomaly-history 45 \
  --z-threshold 2.5 \
  --output custom-scheduler/results/latest_ranking.json
```

## Output

The output JSON includes:

- `hybrid_scheduler` summary metrics
- `prediction_only_scheduler` baseline metrics
- `comparison` deltas (safe-placement rate and anomalous placements)

## Phase 5 Offline Policy Evaluation

Run paired policy comparison using the locked final live policy family defaults:

```bash
python custom-scheduler/offline_policy_evaluation.py \
  --runs-dir ../anomaly-detection/online-telemetry/dataset/runs \
  --model-dir prediction/models \
  --protocol custom-scheduler/evaluation_protocol.json \
  --split-config prediction/results/phase4_validation/locked_predictor_config.json \
  --window 5 \
  --warmup 24 \
  --anomaly-history 45 \
  --anomaly-source nsa \
  --weight-grid 0.9 \
  --z-grid 2.5 \
  --output custom-scheduler/results/offline_policy_evaluation.json
```

This produces:

- Fair paired comparison on identical replay decision points.
- Three policies: prediction-only, anomaly-only, hybrid.
- Metrics: safe placement rate, anomalous placement count/rate, high-contention decision rate, placement fairness (Jain index and percent), decision latency, and protocol utility score.
- Weight telemetry per policy: average effective prediction/anomaly weights, fraction of decisions where anomaly weight exceeds prediction weight, and near-anomaly-only / near-prediction-only fractions.
- A locked offline audit surface that matches the final live policy family by default.
- Final confirmation on untouched test runs.
- Run-level consistency checks for hybrid vs prediction-only on validation and test splits.
- A split manifest JSON saved next to the output for auditability.

If you intentionally want to reopen exploratory offline tuning, pass explicit alternative values for `--anomaly-source`, `--weight-grid`, or `--z-grid` rather than relying on the locked defaults.

## Locked Evaluation Rules

Protocol file: `custom-scheduler/evaluation_protocol.json`

- Locks the selection objective (`utility` or `safe_rate`).
- Configures scheduler behavior (`scheduler.adaptive_weighting`) so hybrid can adapt weights by anomaly risk.
- Defines utility weights across safety, anomaly rate, contention rate, placement fairness, and latency.
- Defines contention threshold used to mark high-contention choices (`contention_relative_cpu_threshold`).
- Freezes held-out test runs for final evaluation.
- Defines consistency requirements for hybrid improvements.
