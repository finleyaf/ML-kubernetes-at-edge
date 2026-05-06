#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
RUN_TAG="${2:-stage_b_redesigned_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULT_DIR="${PROJECT_ROOT}/scheduler-prediction/online/results/${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
OPERATOR_LOG="${LOG_DIR}/operator_console.log"
LATEST_RUN_FILE="${PROJECT_ROOT}/scheduler-prediction/online/results/latest_stage_b_run.txt"

mkdir -p "${LOG_DIR}"

cat > "${LATEST_RUN_FILE}" <<EOF
run_tag=${RUN_TAG}
result_dir=${RESULT_DIR}
operator_log=${OPERATOR_LOG}
stage_b_log=${LOG_DIR}/stage_b.log
summary_json=${RESULT_DIR}/stage_b_summary.json
metadata_file=${RESULT_DIR}/stage_b_metadata.txt
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

{
  echo "== Stage B Operator Log =="
  echo "run_tag=${RUN_TAG}"
  echo "env_file=${ENV_FILE}"
  echo "result_dir=${RESULT_DIR}"
  echo "operator_log=${OPERATOR_LOG}"
  echo "stage_b_log=${LOG_DIR}/stage_b.log"
  echo "summary_json=${RESULT_DIR}/stage_b_summary.json"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
} | tee "${OPERATOR_LOG}"

echo "Tracking progress in ${OPERATOR_LOG}"
echo "Latest run pointer written to ${LATEST_RUN_FILE}"

bash "${SCRIPT_DIR}/run_stage_b_matched_arms.sh" "${ENV_FILE}" "${RUN_TAG}" 2>&1 | tee -a "${OPERATOR_LOG}"