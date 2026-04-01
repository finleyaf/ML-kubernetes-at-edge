#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
RUN_TAG="${2:-stage_b_$(date +%Y%m%d_%H%M%S)}"

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

ZONE="${ZONE:-europe-west2-c}"
CONTROL_NODE="${CONTROL_NODE:-k3s-control}"
K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
DEFAULT_SCHEDULER_NAME="${DEFAULT_SCHEDULER_NAME:-default-scheduler}"
CUSTOM_SCHEDULER_NAME="${CUSTOM_SCHEDULER_NAME:-custom-rank-scheduler}"
STAGE_B_RUNS="${STAGE_B_RUNS:-5}"
STAGE_B_STRESS_NODE="${STAGE_B_STRESS_NODE:-k3s-worker-3}"
STAGE_B_CONTENTION_REL_CPU_THRESHOLD="${STAGE_B_CONTENTION_REL_CPU_THRESHOLD:-1.1}"
STAGE_B_ANOMALY_REL_CPU_THRESHOLD="${STAGE_B_ANOMALY_REL_CPU_THRESHOLD:-1.25}"
VENV_PATH="${VENV_PATH:-.venv}"

WORKLOADS_DIR="${PROJECT_ROOT}/scheduler-prediction/baseline/workloads"
METRIC_SCRIPT="${PROJECT_ROOT}/scheduler-prediction/baseline/evaluation/collect_scheduling_metrics.py"
RESULT_DIR="${PROJECT_ROOT}/scheduler-prediction/online/results/${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
BASELINE_CSV="${RESULT_DIR}/baseline_arm_scheduling.csv"
CUSTOM_CSV="${RESULT_DIR}/custom_arm_scheduling.csv"
SUMMARY_JSON="${RESULT_DIR}/stage_b_summary.json"

mkdir -p "${LOG_DIR}"
rm -f "${BASELINE_CSV}" "${CUSTOM_CSV}"

if [[ -d "${PROJECT_ROOT}/${VENV_PATH}" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/${VENV_PATH}/bin/activate"
fi

run_remote() {
  gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="$*"
}

apply_workload() {
  local workload="$1"
  local pod_name="$2"
  local scheduler_name="$3"

  sed "s/name: .*/name: ${pod_name}/" "${WORKLOADS_DIR}/${workload}.yaml" | \
    awk -v scheduler_name="${scheduler_name}" '
      /^spec:$/ { print; print "  schedulerName: " scheduler_name; next }
      { print }
    ' | gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="cat | kubectl apply -n ${K8S_NAMESPACE} -f -"
}

collect_metrics() {
  local pod_name="$1"
  local output_csv="$2"

  python3 "${METRIC_SCRIPT}" \
    --pod "${pod_name}" \
    --output "${output_csv}" \
    --control-node "${CONTROL_NODE}" \
    --zone "${ZONE}" \
    --contention-relative-cpu-threshold "${STAGE_B_CONTENTION_REL_CPU_THRESHOLD}" \
    --anomaly-relative-cpu-threshold "${STAGE_B_ANOMALY_REL_CPU_THRESHOLD}" 2>&1 | tee -a "${LOG_DIR}/collect_metrics.log"
}

run_arm() {
  local arm_name="$1"
  local scheduler_name="$2"
  local output_csv="$3"

  echo "== Arm: ${arm_name} (schedulerName=${scheduler_name}) ==" | tee -a "${LOG_DIR}/stage_b.log"

  echo "-- Normal phase --" | tee -a "${LOG_DIR}/stage_b.log"
  for i in $(seq 1 "${STAGE_B_RUNS}"); do
    for workload in cpu-pod memory-pod mixed-pod; do
      local pod_name="${workload}-${arm_name}-normal-${i}"
      echo "Deploying ${pod_name}" | tee -a "${LOG_DIR}/stage_b.log"
      apply_workload "${workload}" "${pod_name}" "${scheduler_name}"
      collect_metrics "${pod_name}" "${output_csv}"
      run_remote kubectl wait -n "${K8S_NAMESPACE}" --for=condition=Ready pod/"${pod_name}" --timeout=120s >/dev/null 2>&1 || true
      sleep 4
      run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${pod_name}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
      sleep 2
    done
  done

  echo "-- Stress phase (node ${STAGE_B_STRESS_NODE}) --" | tee -a "${LOG_DIR}/stage_b.log"
  gcloud compute ssh "${STAGE_B_STRESS_NODE}" --zone="${ZONE}" --command="stress --cpu 2 --timeout 240" >/dev/null 2>&1 &
  local stress_pid=$!
  sleep 10

  for i in $(seq 1 "${STAGE_B_RUNS}"); do
    for workload in cpu-pod memory-pod mixed-pod; do
      local pod_name="${workload}-${arm_name}-stress-${i}"
      echo "Deploying ${pod_name}" | tee -a "${LOG_DIR}/stage_b.log"
      apply_workload "${workload}" "${pod_name}" "${scheduler_name}"
      collect_metrics "${pod_name}" "${output_csv}"
      run_remote kubectl wait -n "${K8S_NAMESPACE}" --for=condition=Ready pod/"${pod_name}" --timeout=120s >/dev/null 2>&1 || true
      sleep 4
      run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${pod_name}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
      sleep 2
    done
  done

  wait "${stress_pid}" 2>/dev/null || true
}

echo "== Stage B Matched-Arm Evaluation ==" | tee "${LOG_DIR}/stage_b.log"
echo "tag=${RUN_TAG}" | tee -a "${LOG_DIR}/stage_b.log"
run_arm "baseline" "${DEFAULT_SCHEDULER_NAME}" "${BASELINE_CSV}"
run_arm "custom" "${CUSTOM_SCHEDULER_NAME}" "${CUSTOM_CSV}"

python3 "${PROJECT_ROOT}/scheduler-prediction/online/scripts/analyse_stage_b_results.py" \
  --baseline "${BASELINE_CSV}" \
  --custom "${CUSTOM_CSV}" \
  --output "${SUMMARY_JSON}" | tee -a "${LOG_DIR}/stage_b.log"

cat > "${RESULT_DIR}/stage_b_metadata.txt" <<EOF
run_tag=${RUN_TAG}
namespace=${K8S_NAMESPACE}
default_scheduler_name=${DEFAULT_SCHEDULER_NAME}
custom_scheduler_name=${CUSTOM_SCHEDULER_NAME}
control_node=${CONTROL_NODE}
zone=${ZONE}
runs_per_phase=${STAGE_B_RUNS}
stress_node=${STAGE_B_STRESS_NODE}
contention_rel_cpu_threshold=${STAGE_B_CONTENTION_REL_CPU_THRESHOLD}
anomaly_rel_cpu_threshold=${STAGE_B_ANOMALY_REL_CPU_THRESHOLD}
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "Stage B complete." | tee -a "${LOG_DIR}/stage_b.log"
echo "Summary: ${SUMMARY_JSON}" | tee -a "${LOG_DIR}/stage_b.log"
