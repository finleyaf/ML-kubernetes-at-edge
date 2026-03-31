#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
RUN_TAG="${2:-online_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ ! -f "${PROJECT_ROOT}/${ENV_FILE}" && ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: env file not found: ${ENV_FILE}"
  exit 1
fi

if [[ -f "${PROJECT_ROOT}/${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ROOT}/${ENV_FILE}"
else
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

ZONE="${ZONE:-europe-west2-c}"
INTERVAL="${INTERVAL:-1.0}"
SSH_RETRIES="${SSH_RETRIES:-4}"
SSH_RETRY_DELAY="${SSH_RETRY_DELAY:-20}"
COLLECTOR_STARTUP_TIMEOUT="${COLLECTOR_STARTUP_TIMEOUT:-120}"
RUNS="${RUNS:-6}"
CONTROL_RATIO="${CONTROL_RATIO:-0.25}"
BASELINE="${BASELINE:-120}"
RECOVERY="${RECOVERY:-120}"
SEED="${SEED:-42}"
START_AT="${START_AT:-1}"
LIMIT="${LIMIT:-}"
VENV_PATH="${VENV_PATH:-.venv}"

RESULT_DIR="${PROJECT_ROOT}/scheduler-prediction/online/results/${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${LOG_DIR}"

PLAN_PATH="${PROJECT_ROOT}/anomaly-detection/online-telemetry/dataset/runs/campaign_plan_${RUN_TAG}.json"

echo "== Online Campaign Run =="
echo "tag: ${RUN_TAG}"
echo "plan: ${PLAN_PATH}"
echo "results: ${RESULT_DIR}"

if [[ -d "${PROJECT_ROOT}/${VENV_PATH}" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/${VENV_PATH}/bin/activate"
fi

python3 "${PROJECT_ROOT}/anomaly-detection/online-telemetry/experiments/phase1/generate_campaign_plan.py" \
  --runs "${RUNS}" \
  --control-ratio "${CONTROL_RATIO}" \
  --baseline "${BASELINE}" \
  --recovery "${RECOVERY}" \
  --durations ${DURATIONS:-90 120 150} \
  --seed "${SEED}" \
  --output "${PLAN_PATH}" | tee "${LOG_DIR}/generate_plan.log"

CMD=(
  python3
  "${PROJECT_ROOT}/anomaly-detection/online-telemetry/experiments/phase1/run_campaign.py"
  --plan "${PLAN_PATH}"
  --zone "${ZONE}"
  --interval "${INTERVAL}"
  --ssh-retries "${SSH_RETRIES}"
  --ssh-retry-delay "${SSH_RETRY_DELAY}"
  --collector-startup-timeout "${COLLECTOR_STARTUP_TIMEOUT}"
  --start-at "${START_AT}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/run_campaign.log"

cat > "${RESULT_DIR}/run_metadata.txt" <<EOF
run_tag=${RUN_TAG}
plan_path=${PLAN_PATH}
zone=${ZONE}
interval=${INTERVAL}
ssh_retries=${SSH_RETRIES}
ssh_retry_delay=${SSH_RETRY_DELAY}
collector_startup_timeout=${COLLECTOR_STARTUP_TIMEOUT}
runs=${RUNS}
control_ratio=${CONTROL_RATIO}
baseline=${BASELINE}
recovery=${RECOVERY}
seed=${SEED}
start_at=${START_AT}
limit=${LIMIT}
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo ""
echo "Campaign finished."
echo "Logs: ${LOG_DIR}"
echo "Metadata: ${RESULT_DIR}/run_metadata.txt"
