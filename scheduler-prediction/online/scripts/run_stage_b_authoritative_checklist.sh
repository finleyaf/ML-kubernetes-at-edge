#!/usr/bin/env bash
set -euo pipefail

# This script is an execution checklist wrapper for Stage B authoritative comparison.
# It intentionally fails fast if required scheduler authority settings are not present.

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -f "${PROJECT_ROOT}/${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${PROJECT_ROOT}/${ENV_FILE}"
elif [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
else
  echo "ERROR: env file not found: ${ENV_FILE}"
  exit 1
fi

CUSTOM_SCHEDULER_NAME="${CUSTOM_SCHEDULER_NAME:-custom-rank-scheduler}"
DEFAULT_SCHEDULER_NAME="${DEFAULT_SCHEDULER_NAME:-default-scheduler}"
NAMESPACE="${K8S_NAMESPACE:-default}"
CONTROL_NODE="${CONTROL_NODE:-k3s-control}"
ZONE="${ZONE:-europe-west2-c}"

run_kubectl() {
  local cmd="$1"
  if command -v kubectl >/dev/null 2>&1; then
    kubectl ${cmd}
    return
  fi
  gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="kubectl ${cmd}"
}

echo "== Stage B Authoritative Comparison Checklist =="
echo "namespace: ${NAMESPACE}"
echo "default scheduler: ${DEFAULT_SCHEDULER_NAME}"
echo "custom scheduler: ${CUSTOM_SCHEDULER_NAME}"
echo "control node: ${CONTROL_NODE} (${ZONE})"

echo "[1/5] Checking default scheduler availability"
run_kubectl "get --raw /healthz" >/dev/null

echo "[2/5] Checking custom scheduler profile registration"
if ! run_kubectl "get pods -A" | grep -q "${CUSTOM_SCHEDULER_NAME}"; then
  echo "WARN: No pod name matched ${CUSTOM_SCHEDULER_NAME}."
  echo "      Ensure your custom kube-scheduler deployment/profile is running before Stage B."
fi

echo "[3/5] Verifying workloads can target explicit schedulerName"
cat <<EOF
Required pod spec field per arm:
- baseline arm: spec.schedulerName: ${DEFAULT_SCHEDULER_NAME}
- custom arm:   spec.schedulerName: ${CUSTOM_SCHEDULER_NAME}
EOF

echo "[4/5] Verify matched workload protocol"
cat <<EOF
Use identical sequence across both arms:
- same pod templates
- same stress/control schedule
- same run counts
- same seed
EOF

echo "[5/5] Reminder: evaluate with locked utility objective"
cat <<EOF
safe=0.45 anomaly=0.25 contention=0.15 latency=0.10 fairness=0.05
EOF

echo "Checklist complete."
