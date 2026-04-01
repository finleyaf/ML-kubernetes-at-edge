#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
RUN_TAG="${2:-smoke_$(date +%Y%m%d_%H%M%S)}"

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

MODEL_DIR="${MODEL_DIR:-scheduler-prediction/prediction/models/phase4_weighted_calibrated_expanded_novel_20_safe}"
WINDOW="${WINDOW:-5}"
PRED_WEIGHT="${PRED_WEIGHT:-0.9}"
ANOMALY_WEIGHT="${ANOMALY_WEIGHT:-0.1}"
ANOMALY_HISTORY="${ANOMALY_HISTORY:-45}"
ANOMALY_SOURCE="${ANOMALY_SOURCE:-nsa}"
NSA_NUM_DETECTORS="${NSA_NUM_DETECTORS:-120}"
NSA_RADIUS="${NSA_RADIUS:-0.9}"
KMEANS_THRESHOLD_STD="${KMEANS_THRESHOLD_STD:-2.0}"
Z_THRESHOLD="${Z_THRESHOLD:-2.5}"
VENV_PATH="${VENV_PATH:-.venv}"

RESULT_DIR="${PROJECT_ROOT}/scheduler-prediction/online/results/${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ -d "${PROJECT_ROOT}/${VENV_PATH}" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/${VENV_PATH}/bin/activate"
fi

echo "== Readiness Smoke Test =="

python3 -m py_compile \
  "${PROJECT_ROOT}/scheduler-prediction/custom-scheduler/hybrid_scheduler.py" \
  "${PROJECT_ROOT}/scheduler-prediction/custom-scheduler/rank_live.py"

LATEST_DATASET="$(python3 - <<PY
import pathlib
root = pathlib.Path(r"${PROJECT_ROOT}/anomaly-detection/online-telemetry/dataset/runs")
runs = sorted([p for p in root.glob('run_*') if p.is_dir()], key=lambda p: p.name)
for r in reversed(runs):
    d = r / 'dataset.csv'
    if d.exists() and d.stat().st_size > 0:
        print(d)
        break
PY
)"

if [[ -z "${LATEST_DATASET}" ]]; then
  echo "ERROR: no usable dataset.csv found under anomaly-detection/online-telemetry/dataset/runs"
  exit 1
fi

echo "Using input: ${LATEST_DATASET}"

RANK_OUT="${RESULT_DIR}/latest_ranking.json"
python3 "${PROJECT_ROOT}/scheduler-prediction/custom-scheduler/rank_live.py" \
  --input "${LATEST_DATASET}" \
  --model-dir "${PROJECT_ROOT}/${MODEL_DIR}" \
  --window "${WINDOW}" \
  --pred-weight "${PRED_WEIGHT}" \
  --anomaly-weight "${ANOMALY_WEIGHT}" \
  --anomaly-history "${ANOMALY_HISTORY}" \
  --anomaly-source "${ANOMALY_SOURCE}" \
  --nsa-num-detectors "${NSA_NUM_DETECTORS}" \
  --nsa-radius "${NSA_RADIUS}" \
  --kmeans-threshold-std "${KMEANS_THRESHOLD_STD}" \
  --z-threshold "${Z_THRESHOLD}" \
  --output "${RANK_OUT}" | tee "${LOG_DIR}/rank_live.log"

echo ""
echo "Smoke test complete."
echo "Ranking output: ${RANK_OUT}"
echo "Log: ${LOG_DIR}/rank_live.log"
