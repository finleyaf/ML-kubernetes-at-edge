# Phase 1 Campaign: Data Collection Expansion

This folder implements Phase 1 of the validation-hardening plan:

- Multi-run collection campaigns (e.g., 15-30 runs)
- Configurable control-run ratio (e.g., 20-30%)
- Randomized stress phase order per run
- Varied stress duration and intensity
- Per-run isolated outputs under a unique `run_id`

## Outputs

Each run is stored in:

- `dataset/runs/<run_id>/dataset.csv`
- `dataset/runs/<run_id>/labelled.csv`
- `dataset/runs/<run_id>/phases.json`
- `dataset/runs/<run_id>/metadata.json`

Campaign-level files:

- `dataset/runs/campaign_plan.json`
- `dataset/runs/manifest.json`

## 1) Generate a plan only

```bash
python experiments/phase1/generate_campaign_plan.py \
  --runs 20 \
  --control-ratio 0.25 \
  --baseline 120 \
  --recovery 120 \
  --durations 90 120 150 \
  --seed 42 \
  --output dataset/runs/campaign_plan.json
```

## 2) Execute a plan

```bash
python experiments/phase1/run_campaign.py \
  --plan dataset/runs/campaign_plan.json \
  --zone europe-west2-c \
  --interval 1.0
```

You can tune collector startup validation (default: 12 seconds):

```bash
python experiments/phase1/run_campaign.py \
  --plan dataset/runs/campaign_plan.json \
  --zone europe-west2-c \
  --interval 1.0 \
  --collector-startup-timeout 20
```

Use `--limit` and `--start-at` to resume safely:

```bash
python experiments/phase1/run_campaign.py \
  --plan dataset/runs/campaign_plan.json \
  --zone europe-west2-c \
  --interval 1.0 \
  --start-at 6 \
  --limit 5
```

## 3) One-command helper

```bash
cd anomaly-detection/online-telemetry
RUNS=20 CONTROL_RATIO=0.25 BASELINE=120 RECOVERY=120 \
ZONE=europe-west2-c INTERVAL=1.0 \
./experiments/phase1/run_phase1_campaign.sh
```

Optional env vars:

- `LIMIT` (execute first N runs)
- `START_AT` (1-based run index)
- `SEED` (plan random seed)

## Notes

- Stress runs include CPU, memory, IO, and mixed phases in randomized order.
- Control runs include baseline and recovery only.
- The memory stress target remains the control node, while worker-node-only analysis can still be applied downstream.

## Troubleshooting

If a run fails immediately with a collector import error (for example `ModuleNotFoundError: No module named 'requests'`), install dependencies into the active environment:

```bash
cd anomaly-detection/online-telemetry
source ../../.venv/bin/activate
python -m pip install -r requirements.txt
```

Every run now writes a collector log to:

- `dataset/runs/<run_id>/collector.log`

If startup validation fails, inspect this file first.
