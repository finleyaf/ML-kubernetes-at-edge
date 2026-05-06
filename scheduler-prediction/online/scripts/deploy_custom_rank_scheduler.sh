#!/usr/bin/env bash
set -euo pipefail

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

CONTROL_NODE="${CONTROL_NODE:-k3s-control}"
ZONE="${ZONE:-europe-west2-c}"
CUSTOM_SCHEDULER_NAME="${CUSTOM_SCHEDULER_NAME:-custom-rank-scheduler}"
MANIFEST_REL="${CUSTOM_SCHEDULER_MANIFEST:-scheduler-prediction/online/k8s/custom-rank-scheduler.yaml}"
MANIFEST_PATH="${PROJECT_ROOT}/${MANIFEST_REL}"

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "ERROR: manifest not found: ${MANIFEST_PATH}"
  exit 1
fi

run_kubectl() {
  local cmd="$1"
  if command -v kubectl >/dev/null 2>&1; then
    kubectl ${cmd}
    return
  fi
  gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="kubectl ${cmd}"
}

apply_manifest() {
  if command -v kubectl >/dev/null 2>&1; then
    kubectl apply -f "${MANIFEST_PATH}"
    return
  fi
  cat "${MANIFEST_PATH}" | gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="cat | kubectl apply -f -"
}

echo "== Deploy Custom Scheduler Profile =="
echo "manifest: ${MANIFEST_PATH}"
echo "control node: ${CONTROL_NODE} (${ZONE})"

echo "[1/3] Applying scheduler manifest"
apply_manifest

echo "[2/3] Waiting for deployment rollout"
run_kubectl "rollout status deployment/${CUSTOM_SCHEDULER_NAME} -n kube-system --timeout=240s"

echo "[3/3] Verifying deployment and schedulerName"
run_kubectl "get deployment ${CUSTOM_SCHEDULER_NAME} -n kube-system -o wide"
run_kubectl "get pods -n kube-system -l app=${CUSTOM_SCHEDULER_NAME}"

echo "Deployment complete: ${CUSTOM_SCHEDULER_NAME}"
