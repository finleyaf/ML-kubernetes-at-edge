#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PLAN_PATH="$PROJECT_DIR/dataset/runs/campaign_plan.json"

RUNS="${RUNS:-20}"
CONTROL_RATIO="${CONTROL_RATIO:-0.25}"
BASELINE="${BASELINE:-120}"
RECOVERY="${RECOVERY:-120}"
SEED="${SEED:-42}"
ZONE="${ZONE:-europe-west2-c}"
INTERVAL="${INTERVAL:-1.0}"
LIMIT="${LIMIT:-}"
START_AT="${START_AT:-1}"

echo "Generating campaign plan..."
python "$SCRIPT_DIR/generate_campaign_plan.py" \
  --runs "$RUNS" \
  --control-ratio "$CONTROL_RATIO" \
  --baseline "$BASELINE" \
  --recovery "$RECOVERY" \
  --seed "$SEED" \
  --output "$PLAN_PATH"

echo "Running campaign..."
CMD=(python "$SCRIPT_DIR/run_campaign.py" --plan "$PLAN_PATH" --zone "$ZONE" --interval "$INTERVAL" --start-at "$START_AT")

if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

"${CMD[@]}"

echo "Phase 1 campaign completed."
