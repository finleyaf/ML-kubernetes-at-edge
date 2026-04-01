# Custom Scheduler (Anomaly + Prediction)

This module provides a lightweight hybrid node-ranking engine that combines:

- Predicted future load (from `prediction/predictor.py` models)
- Online anomaly risk (rolling z-score monitor per node)

## Files

- `hybrid_scheduler.py`: core scoring logic
- `simulate_scheduler.py`: replay-based evaluation on labelled telemetry
- `rank_live.py`: rank nodes from recent telemetry and return best target node
- `offline_policy_evaluation.py`: paired offline comparison for prediction-only, anomaly-only, and hybrid policies with validation-only tuning and untouched test evaluation

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

Run paired policy comparison with validation-only grid search and untouched holdout test:

```bash
python custom-scheduler/offline_policy_evaluation.py \
  --runs-dir ../anomaly-detection/online-telemetry/dataset/runs \
  --model-dir prediction/models \
  --protocol custom-scheduler/evaluation_protocol.json \
  --split-config prediction/results/phase4_validation/locked_predictor_config.json \
  --stage selection \
  --window 5 \
  --warmup 15 \
  --anomaly-history 45 \
  --weight-grid 0.5,0.6,0.7,0.8,0.9 \
  --z-grid 2.0,2.5,3.0,3.5 \
  --output custom-scheduler/results/offline_policy_evaluation.json
```

This produces:

- Fair paired comparison on identical replay decision points.
- Three policies: prediction-only, anomaly-only, hybrid.
- Metrics: safe placement rate, anomalous placement count/rate, high-contention decision rate, placement fairness (Jain index and percent), decision latency, and protocol utility score.
- Weight telemetry per policy: average effective prediction/anomaly weights, fraction of decisions where anomaly weight exceeds prediction weight, and near-anomaly-only / near-prediction-only fractions.
- Grid search on validation runs only.
- Final confirmation on untouched test runs.
- Run-level consistency checks for hybrid vs prediction-only on validation and test splits.
- A split manifest JSON saved next to the output for auditability.

## Anti-Peeking Workflow (Recommended)

Use two explicit stages to prevent accidental holdout peeking during iterative tuning:

1. Selection stage (dev runs only, no holdout metrics written)

```bash
python custom-scheduler/offline_policy_evaluation.py \
  --runs-dir ../anomaly-detection/online-telemetry/dataset/runs \
  --model-dir prediction/models \
  --protocol custom-scheduler/evaluation_protocol.json \
  --split-config prediction/results/phase4_validation/locked_predictor_config.json \
  --stage selection \
  --weight-grid 0.5,0.6,0.7,0.8,0.9 \
  --z-grid 2.0,2.5,3.0,3.5 \
  --output custom-scheduler/results/offline_policy_selection.json
```

2. Audit stage (strict holdout only, fixed config from selection output)

```bash
python custom-scheduler/offline_policy_evaluation.py \
  --runs-dir ../anomaly-detection/online-telemetry/dataset/runs \
  --model-dir prediction/models \
  --protocol custom-scheduler/evaluation_protocol.json \
  --split-config prediction/results/phase4_validation/locked_predictor_config.json \
  --stage audit \
  --selected-config custom-scheduler/results/offline_policy_selection.json \
  --output custom-scheduler/results/offline_policy_audit.json
```

Notes:

- `--stage selection` never computes or emits holdout evaluation results.
- `--stage audit` requires either `--selected-config` or both `--fixed-pred-weight` and `--fixed-z-threshold`.
- `--stage both` is available for convenience but not recommended for iterative experimentation.

## Locked Evaluation Rules

Protocol file: `custom-scheduler/evaluation_protocol.json`

- Locks the selection objective (`utility` or `safe_rate`).
- Configures scheduler behavior (`scheduler.adaptive_weighting`) so hybrid can adapt weights by anomaly risk.
- Defines utility weights across safety, anomaly rate, contention rate, placement fairness, and latency.
- Defines contention threshold used to mark high-contention choices (`contention_relative_cpu_threshold`).
- Freezes held-out test runs for final evaluation.
- Defines consistency requirements for hybrid improvements.
