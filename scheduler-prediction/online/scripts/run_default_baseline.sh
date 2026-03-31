#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
RUN_TAG="${2:-baseline_$(date +%Y%m%d_%H%M%S)}"

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

RESULT_DIR="${PROJECT_ROOT}/scheduler-prediction/online/results/${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${LOG_DIR}"

VENV_PATH="${VENV_PATH:-.venv}"
if [[ -d "${PROJECT_ROOT}/${VENV_PATH}" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/${VENV_PATH}/bin/activate"
fi

echo "== Default Scheduler Baseline Run =="

bash "${PROJECT_ROOT}/scheduler-prediction/baseline/run_baseline_eval.sh" 2>&1 | tee "${LOG_DIR}/baseline_eval.log"

BASE_RESULTS="${PROJECT_ROOT}/scheduler-prediction/baseline/evaluation/results"
cp -f "${BASE_RESULTS}/baseline_scheduling.csv" "${RESULT_DIR}/baseline_scheduling.csv"
cp -f "${BASE_RESULTS}/baseline_analysis.json" "${RESULT_DIR}/baseline_analysis.json"

echo ""
echo "Baseline complete."
echo "Copied artifacts to: ${RESULT_DIR}"
echo "Log: ${LOG_DIR}/baseline_eval.log"
