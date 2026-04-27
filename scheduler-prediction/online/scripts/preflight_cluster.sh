#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
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
CONTROL_NODE="${CONTROL_NODE:-k3s-control}"
NODES="${NODES:-k3s-control k3s-worker-2 k3s-worker-3 k3s-worker-4 raspberrypi}"
NETDATA_BASE_URL="${NETDATA_BASE_URL:-http://localhost:20000}"
VENV_PATH="${VENV_PATH:-.venv}"

echo "== Preflight: local toolchain =="
for cmd in python3 gcloud curl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "MISSING: ${cmd}"
    exit 1
  fi
  echo "OK: ${cmd}"
done

echo "== Preflight: gcloud auth =="
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
  echo "ERROR: no active gcloud account. Run: gcloud auth login"
  exit 1
fi
echo "OK: active gcloud account: ${ACTIVE_ACCOUNT}"

echo "== Preflight: netdata endpoint =="
if curl -fsS --max-time 6 "${NETDATA_BASE_URL%/}/api/v1/info" >/dev/null; then
  echo "OK: ${NETDATA_BASE_URL%/}/api/v1/info reachable"
else
  echo "WARN: netdata endpoint unreachable: ${NETDATA_BASE_URL%/}/api/v1/info"
fi

echo "== Preflight: SSH and node prerequisites =="
for node in ${NODES}; do
  echo "-- checking ${node}"
  gcloud compute ssh "${node}" --zone="${ZONE}" --command="echo ok" >/dev/null
  gcloud compute ssh "${node}" --zone="${ZONE}" --command="command -v stress >/dev/null && echo stress_ok || (echo stress_missing; exit 1)" >/dev/null
  echo "OK: ${node}"
done

echo "== Preflight: control node kubectl =="
gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="kubectl version --client >/dev/null && echo kubectl_ok" >/dev/null
echo "OK: kubectl available on ${CONTROL_NODE}"

echo "== Preflight: repository files =="
REQUIRED_FILES=(
  "anomaly-detection/online-telemetry/experiments/phase1/generate_campaign_plan.py"
  "anomaly-detection/online-telemetry/experiments/phase1/run_campaign.py"
  "scheduler-prediction/custom-scheduler/rank_live.py"
)
for rel in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${PROJECT_ROOT}/${rel}" ]]; then
    echo "ERROR: missing required file: ${rel}"
    exit 1
  fi
  echo "OK: ${rel}"
done

echo "== Preflight: python environment =="
if [[ -d "${PROJECT_ROOT}/${VENV_PATH}" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/${VENV_PATH}/bin/activate"
fi
python3 - <<'PY'
mods = ["pandas", "numpy", "sklearn", "requests", "joblib"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print("MISSING_PY_MODULES", ",".join(missing))
    raise SystemExit(1)
print("OK: python modules present")
PY

echo ""
echo "Preflight complete. You are ready to run online campaign and smoke tests."
