#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scheduler-prediction/online/config/online_test.env}"
RUN_TAG="${2:-stage_b_redesigned_$(date +%Y%m%d_%H%M%S)}"

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

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    echo "$path"
  else
    echo "${PROJECT_ROOT}/${path}"
  fi
}

ZONE="${ZONE:-europe-west2-c}"
CONTROL_NODE="${CONTROL_NODE:-k3s-control}"
K8S_NAMESPACE="${K8S_NAMESPACE:-default}"
DEFAULT_SCHEDULER_NAME="${DEFAULT_SCHEDULER_NAME:-default-scheduler}"
CUSTOM_SCHEDULER_NAME="${CUSTOM_SCHEDULER_NAME:-custom-rank-scheduler}"
BASELINE_DECISION_MODE="${BASELINE_DECISION_MODE:-matched_round_robin_and_pin}"
WORKER_NODES="${WORKER_NODES:-k3s-worker-2 k3s-worker-3 k3s-worker-4 raspberrypi}"
# Stage B should always collect from the control plane plus the currently eligible worker pool.
NODES="${CONTROL_NODE} ${WORKER_NODES}"
NETDATA_BASE_URL="${NETDATA_BASE_URL:-http://localhost:20000}"
STAGE_B_RUNS="${STAGE_B_RUNS:-10}"
STAGE_B_WORKLOADS="${STAGE_B_WORKLOADS:-cpu-pod memory-pod mixed-pod}"
STAGE_B_ARM_ORDER_MODE="${STAGE_B_ARM_ORDER_MODE:-counterbalanced}"
PI_NODE_NAME="${PI_NODE_NAME:-}"
PI_ELIGIBLE_WORKLOADS="${PI_ELIGIBLE_WORKLOADS:-}"
STAGE_B_WARMUP_SAMPLES="${STAGE_B_WARMUP_SAMPLES:-24}"
STAGE_B_WARMUP_INTERVAL="${STAGE_B_WARMUP_INTERVAL:-1.0}"
STAGE_B_WASHOUT_SECONDS="${STAGE_B_WASHOUT_SECONDS:-8}"
STAGE_B_STRESS_PROFILES="${STAGE_B_STRESS_PROFILES:-cpu memory mixed}"
STAGE_B_STRESS_TARGETS="${STAGE_B_STRESS_TARGETS:-${WORKER_NODES}}"
STAGE_B_STRESS_TIMEOUT="${STAGE_B_STRESS_TIMEOUT:-420}"
STAGE_B_STRESS_STARTUP_SECONDS="${STAGE_B_STRESS_STARTUP_SECONDS:-10}"
STAGE_B_STRESS_IMAGE="${STAGE_B_STRESS_IMAGE:-alpine:3.20}"
STAGE_B_STRESS_READY_TIMEOUT_SECONDS="${STAGE_B_STRESS_READY_TIMEOUT_SECONDS:-120}"
STAGE_B_STRESS_CPU_WORKERS="${STAGE_B_STRESS_CPU_WORKERS:-2}"
STAGE_B_STRESS_RESERVED_CPUS="${STAGE_B_STRESS_RESERVED_CPUS:-1}"
STAGE_B_STRESS_MEMORY_MIB="${STAGE_B_STRESS_MEMORY_MIB:-512}"
STAGE_B_STRESS_MIXED_MEMORY_MIB="${STAGE_B_STRESS_MIXED_MEMORY_MIB:-256}"
STAGE_B_STRESS_RESERVED_MEMORY_MIB="${STAGE_B_STRESS_RESERVED_MEMORY_MIB:-768}"
STAGE_B_STRESS_MIN_MEMORY_MIB="${STAGE_B_STRESS_MIN_MEMORY_MIB:-128}"
STAGE_B_STRESS_NICE_LEVEL="${STAGE_B_STRESS_NICE_LEVEL:-10}"
SNAPSHOT_MAX_ATTEMPTS="${SNAPSHOT_MAX_ATTEMPTS:-5}"
SNAPSHOT_RETRY_DELAY_SECONDS="${SNAPSHOT_RETRY_DELAY_SECONDS:-2}"
STAGE_B_CONTENTION_REL_CPU_THRESHOLD="${STAGE_B_CONTENTION_REL_CPU_THRESHOLD:-1.1}"
STAGE_B_ANOMALY_REL_CPU_THRESHOLD="${STAGE_B_ANOMALY_REL_CPU_THRESHOLD:-1.25}"
MODEL_DIR_PATH="$(resolve_path "${MODEL_DIR:-scheduler-prediction/prediction/models/phase4_weighted_calibrated_expanded_novel_20_safe}")"
WINDOW="${WINDOW:-5}"
PRED_WEIGHT="${PRED_WEIGHT:-0.9}"
ANOMALY_WEIGHT="${ANOMALY_WEIGHT:-0.1}"
ANOMALY_HISTORY="${ANOMALY_HISTORY:-45}"
ANOMALY_SOURCE="${ANOMALY_SOURCE:-nsa}"
NSA_NUM_DETECTORS="${NSA_NUM_DETECTORS:-120}"
NSA_RADIUS="${NSA_RADIUS:-0.9}"
KMEANS_THRESHOLD_STD="${KMEANS_THRESHOLD_STD:-2.0}"
Z_THRESHOLD="${Z_THRESHOLD:-2.5}"
ADAPTIVE_WEIGHTING="${ADAPTIVE_WEIGHTING:-false}"
ADAPTIVE_RISK_LOW="${ADAPTIVE_RISK_LOW:-0.2}"
ADAPTIVE_RISK_HIGH="${ADAPTIVE_RISK_HIGH:-0.7}"
ADAPTIVE_MAX_SHIFT="${ADAPTIVE_MAX_SHIFT:-0.35}"
ADAPTIVE_MIN_PREDICTION_WEIGHT="${ADAPTIVE_MIN_PREDICTION_WEIGHT:-0.05}"
ADAPTIVE_MAX_PREDICTION_WEIGHT="${ADAPTIVE_MAX_PREDICTION_WEIGHT:-0.95}"
CONTENTION_RELATIVE_CPU_THRESHOLD="${CONTENTION_RELATIVE_CPU_THRESHOLD:-1.1}"
CONTENTION_PENALTY_FACTOR="${CONTENTION_PENALTY_FACTOR:-0.0}"
SAFE_RELATIVE_CPU_THRESHOLD="${SAFE_RELATIVE_CPU_THRESHOLD:-1.1}"
UNSAFE_PENALTY_FACTOR="${UNSAFE_PENALTY_FACTOR:-0.0}"
AVOID_UNSAFE_NODES="${AVOID_UNSAFE_NODES:-false}"
VENV_PATH="${VENV_PATH:-.venv}"

WORKLOADS_DIR="${PROJECT_ROOT}/scheduler-prediction/baseline/workloads"
METRIC_SCRIPT="${PROJECT_ROOT}/scheduler-prediction/baseline/evaluation/collect_scheduling_metrics.py"
SNAPSHOT_COLLECTOR="${PROJECT_ROOT}/anomaly-detection/online-telemetry/data_collection/netdata_collector.py"
CAPACITY_EXPORTER="${SCRIPT_DIR}/export_node_capacities.py"
RANK_SCRIPT="${PROJECT_ROOT}/scheduler-prediction/custom-scheduler/rank_live.py"
ANALYSE_SCRIPT="${SCRIPT_DIR}/analyse_stage_b_redesigned_results.py"
RESULT_DIR="${PROJECT_ROOT}/scheduler-prediction/online/results/${RUN_TAG}"
LOG_DIR="${RESULT_DIR}/logs"
ARTIFACT_DIR="${RESULT_DIR}/decision_artifacts"
CUSTOM_HISTORY="${ARTIFACT_DIR}/custom_history.csv"
CAPACITY_JSON="${RESULT_DIR}/node_capacities.json"
BASELINE_CSV="${RESULT_DIR}/baseline_arm_scheduling.csv"
CUSTOM_CSV="${RESULT_DIR}/custom_arm_scheduling.csv"
SUMMARY_JSON="${RESULT_DIR}/stage_b_summary.json"

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}/baseline" "${ARTIFACT_DIR}/custom"
rm -f "${BASELINE_CSV}" "${CUSTOM_CSV}" "${CUSTOM_HISTORY}" "${CAPACITY_JSON}"

if [[ -d "${PROJECT_ROOT}/${VENV_PATH}" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/${VENV_PATH}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

read -r -a NODE_ARRAY <<< "${NODES}"
read -r -a WORKER_ARRAY <<< "${WORKER_NODES}"
read -r -a WORKLOAD_ARRAY <<< "${STAGE_B_WORKLOADS}"
read -r -a PI_ELIGIBLE_WORKLOAD_ARRAY <<< "${PI_ELIGIBLE_WORKLOADS}"
read -r -a STRESS_PROFILE_ARRAY <<< "${STAGE_B_STRESS_PROFILES}"
read -r -a STRESS_TARGET_ARRAY <<< "${STAGE_B_STRESS_TARGETS}"

run_remote() {
  gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="$*"
}

manifest_path() {
  echo "${WORKLOADS_DIR}/$1.yaml"
}

non_pi_worker_nodes() {
  local node

  for node in "${WORKER_ARRAY[@]}"; do
    if [[ -n "${PI_NODE_NAME}" && "${node}" == "${PI_NODE_NAME}" ]]; then
      continue
    fi
    printf '%s\n' "${node}"
  done
}

workload_allows_pi() {
  local workload="$1"
  local candidate

  if [[ -z "${PI_NODE_NAME}" || ${#PI_ELIGIBLE_WORKLOAD_ARRAY[@]} -eq 0 ]]; then
    return 1
  fi

  for candidate in "${PI_ELIGIBLE_WORKLOAD_ARRAY[@]}"; do
    if [[ "${candidate}" == "${workload}" ]]; then
      return 0
    fi
  done

  return 1
}

eligible_nodes_for_workload() {
  local workload="$1"
  local -a nodes=()
  local node

  if workload_allows_pi "${workload}"; then
    nodes=("${WORKER_ARRAY[@]}")
  else
    while IFS= read -r node; do
      if [[ -n "${node}" ]]; then
        nodes+=("${node}")
      fi
    done < <(non_pi_worker_nodes)

    if (( ${#nodes[@]} == 0 )); then
      nodes=("${WORKER_ARRAY[@]}")
    fi
  fi

  printf '%s\n' "${nodes[*]}"
}

extract_request_value() {
  local workload="$1"
  local resource="$2"
  awk -v resource="${resource}" '
    $1 == "requests:" { in_requests = 1; next }
    in_requests && $1 == "limits:" { in_requests = 0 }
    in_requests && $1 == resource ":" {
      gsub(/"/, "", $2)
      print $2
      exit
    }
  ' "$(manifest_path "${workload}")"
}

cpu_to_millicores() {
  local raw="$1"
  if [[ -z "${raw}" ]]; then
    echo "0"
  elif [[ "${raw}" == *m ]]; then
    echo "${raw%m}"
  else
    awk -v value="${raw}" 'BEGIN { printf "%.4f", value * 1000.0 }'
  fi
}

memory_to_mib() {
  local raw="$1"
  case "${raw}" in
    *Ki)
      awk -v value="${raw%Ki}" 'BEGIN { printf "%.4f", value / 1024.0 }'
      ;;
    *Mi)
      echo "${raw%Mi}"
      ;;
    *Gi)
      awk -v value="${raw%Gi}" 'BEGIN { printf "%.4f", value * 1024.0 }'
      ;;
    *Ti)
      awk -v value="${raw%Ti}" 'BEGIN { printf "%.4f", value * 1048576.0 }'
      ;;
    "")
      echo "0"
      ;;
    *)
      echo "${raw}"
      ;;
  esac
}

workload_cpu_millicores() {
  cpu_to_millicores "$(extract_request_value "$1" "cpu")"
}

workload_memory_mib() {
  memory_to_mib "$(extract_request_value "$1" "memory")"
}

render_workload_manifest() {
  local workload="$1"
  local pod_name="$2"
  local scheduler_name="$3"
  local node_name="$4"
  local eligible_nodes="$5"

  sed "s/name: .*/name: ${pod_name}/" "$(manifest_path "${workload}")" | awk \
    -v scheduler_name="${scheduler_name}" \
    -v node_name="${node_name}" \
    -v eligible_nodes="${eligible_nodes}" '
      /^spec:$/ {
        print
        if (scheduler_name != "") {
          print "  schedulerName: " scheduler_name
        }
        if (node_name != "") {
          print "  nodeName: " node_name
        }
        if (eligible_nodes != "") {
          split(eligible_nodes, nodes, /[[:space:]]+/)
          print "  affinity:"
          print "    nodeAffinity:"
          print "      requiredDuringSchedulingIgnoredDuringExecution:"
          print "        nodeSelectorTerms:"
          print "          - matchExpressions:"
          print "              - key: kubernetes.io/hostname"
          print "                operator: In"
          print "                values:"
          for (i in nodes) {
            if (nodes[i] != "") {
              print "                  - " nodes[i]
            }
          }
        }
        next
      }
      { print }
    '
}

apply_workload() {
  local workload="$1"
  local pod_name="$2"
  local scheduler_name="$3"
  local node_name="$4"
  local eligible_nodes="$5"

  # A previous interrupted run can leave a pod with the same deterministic name.
  # Delete it first so reruns do not fail on immutable scheduling fields.
  cleanup_pod "${pod_name}"

  render_workload_manifest "${workload}" "${pod_name}" "${scheduler_name}" "${node_name}" "${eligible_nodes}" | \
    gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="cat | kubectl apply -n ${K8S_NAMESPACE} -f -"
}

cleanup_pod() {
  local pod_name="$1"
  run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${pod_name}" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  run_remote kubectl wait -n "${K8S_NAMESPACE}" --for=delete pod/"${pod_name}" --timeout=120s >/dev/null 2>&1 || true
}

sanitize_k8s_name() {
  local raw="$1"
  raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9.-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [[ -z "${raw}" ]]; then
    echo "ERROR: unable to derive a valid Kubernetes name from '${1}'" >&2
    exit 1
  fi
  echo "${raw}"
}

validate_snapshot_workers() {
  local snapshot_csv="$1"
  local context_label="$2"
  "${PYTHON_BIN}" - "${snapshot_csv}" "${context_label}" "${WORKER_ARRAY[@]}" <<'PY'
import csv
import sys

snapshot_path = sys.argv[1]
context = sys.argv[2]
required_workers = sys.argv[3:]

with open(snapshot_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

present_workers = {row.get("node", "").strip() for row in rows if row.get("node")}
missing_workers = [node for node in required_workers if node not in present_workers]

if missing_workers:
    present = ", ".join(sorted(worker for worker in present_workers if worker)) or "<none>"
    sys.stderr.write(
        f"ERROR: telemetry snapshot '{context}' is missing required worker nodes: {', '.join(missing_workers)}\n"
    )
    sys.stderr.write(f"Present worker rows: {present}\n")
    raise SystemExit(1)
PY
}

merge_snapshot_rows() {
  local destination_csv="$1"
  local source_csv="$2"
  "${PYTHON_BIN}" - "${destination_csv}" "${source_csv}" "${NODE_ARRAY[@]}" <<'PY'
import csv
import os
import sys

destination = sys.argv[1]
source = sys.argv[2]
node_order = sys.argv[3:]

rows_by_node = {}
header = None

for path in (destination, source):
  if not os.path.exists(path):
    continue
  with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames:
      header = reader.fieldnames
    for row in reader:
      node = row.get("node", "").strip()
      if node:
        rows_by_node[node] = row

if not header:
  raise SystemExit(0)

ordered_nodes = []
seen_nodes = set()
for node in node_order:
  if node in rows_by_node:
    ordered_nodes.append(node)
    seen_nodes.add(node)
for node in rows_by_node:
  if node not in seen_nodes:
    ordered_nodes.append(node)

with open(destination, "w", newline="", encoding="utf-8") as handle:
  writer = csv.DictWriter(handle, fieldnames=header)
  writer.writeheader()
  for node in ordered_nodes:
    writer.writerow(rows_by_node[node])
PY
}

capture_snapshot() {
  local output_csv="$1"
  local context_label="$2"
  local attempt
  local attempt_csv
  local aggregate_csv="${output_csv%.csv}.aggregate.csv"

  rm -f "${output_csv}" "${aggregate_csv}"

  for (( attempt=1; attempt<=SNAPSHOT_MAX_ATTEMPTS; attempt++ )); do
    attempt_csv="${output_csv%.csv}.attempt${attempt}.csv"
    rm -f "${attempt_csv}"

    "${PYTHON_BIN}" "${SNAPSHOT_COLLECTOR}" \
      --base-url "${NETDATA_BASE_URL}" \
      --output "${attempt_csv}" \
      --interval "${STAGE_B_WARMUP_INTERVAL}" \
      --samples 1 \
      --nodes "${NODE_ARRAY[@]}" >> "${LOG_DIR}/snapshot.log" 2>&1

    merge_snapshot_rows "${aggregate_csv}" "${attempt_csv}"
    rm -f "${attempt_csv}"

    if validate_snapshot_workers "${aggregate_csv}" "${context_label}" >> "${LOG_DIR}/snapshot.log" 2>&1; then
      mv "${aggregate_csv}" "${output_csv}"
      return 0
    fi

    if (( attempt < SNAPSHOT_MAX_ATTEMPTS )); then
      echo "Retrying snapshot capture for ${context_label} (attempt ${attempt}/${SNAPSHOT_MAX_ATTEMPTS}) after incomplete worker coverage in aggregate snapshot" | tee -a "${LOG_DIR}/snapshot.log"
      sleep "${SNAPSHOT_RETRY_DELAY_SECONDS}"
      continue
    fi
  done

  mv "${aggregate_csv}" "${output_csv}"
  validate_snapshot_workers "${output_csv}" "${context_label}"
}

seed_custom_history_if_needed() {
  if [[ -f "${CUSTOM_HISTORY}" ]] && [[ $(wc -l < "${CUSTOM_HISTORY}") -gt 1 ]]; then
    return
  fi

  echo "Seeding custom-arm telemetry history (${STAGE_B_WARMUP_SAMPLES} samples)" | tee -a "${LOG_DIR}/stage_b.log"
  "${PYTHON_BIN}" "${SNAPSHOT_COLLECTOR}" \
    --base-url "${NETDATA_BASE_URL}" \
    --output "${CUSTOM_HISTORY}" \
    --interval "${STAGE_B_WARMUP_INTERVAL}" \
    --samples "${STAGE_B_WARMUP_SAMPLES}" \
    --nodes "${NODE_ARRAY[@]}" | tee -a "${LOG_DIR}/snapshot.log"
  validate_snapshot_workers "${CUSTOM_HISTORY}" "custom_history_seed"
}

append_snapshot_to_history() {
  local snapshot_csv="$1"
  if [[ -f "${CUSTOM_HISTORY}" ]] && [[ $(wc -l < "${CUSTOM_HISTORY}") -gt 0 ]]; then
    tail -n +2 "${snapshot_csv}" >> "${CUSTOM_HISTORY}"
  else
    cp "${snapshot_csv}" "${CUSTOM_HISTORY}"
  fi
}

export_node_capacities() {
  "${PYTHON_BIN}" "${CAPACITY_EXPORTER}" \
    --control-node "${CONTROL_NODE}" \
    --zone "${ZONE}" \
    --output "${CAPACITY_JSON}" \
    --nodes "${WORKER_ARRAY[@]}" | tee -a "${LOG_DIR}/stage_b.log"
}

parse_ranking_fields() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = json.load(f)

best = payload["ranking"][0]
values = [
    payload.get("best_node", ""),
    best.get("total_score", ""),
    best.get("predicted_load", ""),
    best.get("base_predicted_load", ""),
    best.get("anomaly_risk", ""),
    best.get("cpu_request_fraction", ""),
    best.get("memory_request_fraction", ""),
    best.get("capacity_penalty", ""),
    best.get("prediction_source", ""),
]
print("\t".join("" if value is None else str(value) for value in values))
PY
}

arm_order_for_trial() {
  local trial="$1"
  case "${STAGE_B_ARM_ORDER_MODE}" in
    baseline-first)
      echo "baseline custom"
      ;;
    custom-first)
      echo "custom baseline"
      ;;
    counterbalanced)
      if (( trial % 2 == 1 )); then
        echo "baseline custom"
      else
        echo "custom baseline"
      fi
      ;;
    *)
      echo "ERROR: unsupported STAGE_B_ARM_ORDER_MODE=${STAGE_B_ARM_ORDER_MODE}" >&2
      exit 1
      ;;
  esac
}

baseline_target_for() {
  local trial="$1"
  local phase_name="$2"
  local workload="$3"
  local eligible_nodes="${4:-${WORKER_NODES}}"
  local -a eligible_array=()
  local phase_index=0
  local workload_index=-1
  local phases_per_trial
  local decisions_per_trial
  local decision_index
  local i

  read -r -a eligible_array <<< "${eligible_nodes}"

  if (( ${#eligible_array[@]} == 0 )); then
    echo "ERROR: WORKER_NODES must define at least one eligible worker for baseline placement" >&2
    exit 1
  fi

  case "${phase_name}" in
    normal)
      phase_index=0
      ;;
    stress_*)
      phase_index=-1
      local stress_profile_name="${phase_name#stress_}"
      for i in "${!STRESS_PROFILE_ARRAY[@]}"; do
        if [[ "${STRESS_PROFILE_ARRAY[$i]}" == "${stress_profile_name}" ]]; then
          phase_index=$(( i + 1 ))
          break
        fi
      done
      if (( phase_index < 0 )); then
        echo "ERROR: unsupported stress phase '${phase_name}' for baseline targeting" >&2
        exit 1
      fi
      ;;
    *)
      echo "ERROR: unsupported phase '${phase_name}' for baseline targeting" >&2
      exit 1
      ;;
  esac

  for i in "${!WORKLOAD_ARRAY[@]}"; do
    if [[ "${WORKLOAD_ARRAY[$i]}" == "${workload}" ]]; then
      workload_index="${i}"
      break
    fi
  done

  if (( workload_index < 0 )); then
    echo "ERROR: unsupported workload '${workload}' for baseline targeting" >&2
    exit 1
  fi

  phases_per_trial=$(( ${#STRESS_PROFILE_ARRAY[@]} + 1 ))
  decisions_per_trial=$(( phases_per_trial * ${#WORKLOAD_ARRAY[@]} ))
  decision_index=$(( ((trial - 1) * decisions_per_trial) + (phase_index * ${#WORKLOAD_ARRAY[@]}) + workload_index ))

  echo "${eligible_array[$(( decision_index % ${#eligible_array[@]} ))]}"
}

stress_cpu_workers_for_target() {
  local target="$1"
  local allocatable_cpu
  local max_workers

  allocatable_cpu="$(run_remote kubectl get node "${target}" -o jsonpath='{.status.allocatable.cpu}' 2>/dev/null || true)"
  if [[ ! "${allocatable_cpu}" =~ ^[0-9]+$ ]]; then
    echo "${STAGE_B_STRESS_CPU_WORKERS}"
    return
  fi

  max_workers=$(( allocatable_cpu - STAGE_B_STRESS_RESERVED_CPUS ))
  if (( max_workers < 1 )); then
    max_workers=1
  fi
  if (( max_workers > STAGE_B_STRESS_CPU_WORKERS )); then
    max_workers="${STAGE_B_STRESS_CPU_WORKERS}"
  fi

  echo "${max_workers}"
}

parse_memory_to_mib() {
  local raw_value="$1"

  "${PYTHON_BIN}" - "${raw_value}" <<'PY'
import sys

value = sys.argv[1].strip()
units = {
    "Ki": 1.0 / 1024.0,
    "Mi": 1.0,
    "Gi": 1024.0,
    "Ti": 1024.0 * 1024.0,
}

for suffix, scale in units.items():
    if value.endswith(suffix):
        print(max(1, int(float(value[: -len(suffix)]) * scale)))
        break
else:
    print(max(1, int(float(value))))
PY
}

stress_memory_mib_for_target() {
  local profile="$1"
  local target="$2"
  local default_mib
  local allocatable_raw
  local allocatable_mib
  local max_mib

  case "${profile}" in
    memory)
      default_mib="${STAGE_B_STRESS_MEMORY_MIB}"
      ;;
    mixed)
      default_mib="${STAGE_B_STRESS_MIXED_MEMORY_MIB}"
      ;;
    *)
      echo "0"
      return
      ;;
  esac

  allocatable_raw="$(run_remote kubectl get node "${target}" -o jsonpath='{.status.allocatable.memory}' 2>/dev/null || true)"
  if [[ -z "${allocatable_raw}" ]]; then
    echo "${default_mib}"
    return
  fi

  allocatable_mib="$(parse_memory_to_mib "${allocatable_raw}" 2>/dev/null || true)"
  if [[ ! "${allocatable_mib}" =~ ^[0-9]+$ ]]; then
    echo "${default_mib}"
    return
  fi

  max_mib=$(( allocatable_mib - STAGE_B_STRESS_RESERVED_MEMORY_MIB ))
  if (( max_mib < STAGE_B_STRESS_MIN_MEMORY_MIB )); then
    max_mib="${STAGE_B_STRESS_MIN_MEMORY_MIB}"
  fi
  if (( max_mib < default_mib )); then
    echo "${max_mib}"
    return
  fi

  echo "${default_mib}"
}

stress_shell_command_for_profile() {
  local profile="$1"
  local target="$2"
  local cmd
  local install_cmd

  cmd="$(stress_command_for_profile "${profile}" "${target}")"
  install_cmd='if ! command -v stress-ng >/dev/null 2>&1; then if command -v apk >/dev/null 2>&1; then apk add --no-cache stress-ng >/dev/null; else echo "stress-ng is unavailable and apk is not installed" >&2; exit 127; fi; fi;'
  printf '%s if command -v nice >/dev/null 2>&1; then exec nice -n %s %s; else exec %s; fi' \
    "${install_cmd}" "${STAGE_B_STRESS_NICE_LEVEL}" "${cmd}" "${cmd}"
}

stress_command_for_profile() {
  local profile="$1"
  local target="$2"
  local cpu_workers
  local memory_mib
  case "${profile}" in
    cpu)
      cpu_workers="$(stress_cpu_workers_for_target "${target}")"
      echo "stress-ng --cpu ${cpu_workers} --timeout ${STAGE_B_STRESS_TIMEOUT}s"
      ;;
    memory)
      memory_mib="$(stress_memory_mib_for_target "${profile}" "${target}")"
      echo "stress-ng --vm 1 --vm-bytes ${memory_mib}M --vm-keep --timeout ${STAGE_B_STRESS_TIMEOUT}s"
      ;;
    mixed)
      memory_mib="$(stress_memory_mib_for_target "${profile}" "${target}")"
      echo "stress-ng --cpu 1 --vm 1 --vm-bytes ${memory_mib}M --vm-keep --timeout ${STAGE_B_STRESS_TIMEOUT}s"
      ;;
    *)
      echo "ERROR: unsupported stress profile ${profile}" >&2
      exit 1
      ;;
  esac
}

pick_stress_target() {
  local trial="$1"
  local profile_index="$2"
  local idx=$(( (trial + profile_index - 2) % ${#STRESS_TARGET_ARRAY[@]} ))
  echo "${STRESS_TARGET_ARRAY[$idx]}"
}

start_stress() {
  local profile="$1"
  local target="$2"
  local cmd
  local stress_pod_name

  cmd="$(stress_shell_command_for_profile "${profile}" "${target}")"
  stress_pod_name="$(sanitize_k8s_name "stage-b-stress-${profile}-${target}-${RUN_TAG}")"

  run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${stress_pod_name}" --ignore-not-found --wait=false >> "${LOG_DIR}/stress.log" 2>&1 || true

  cat <<EOF | gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="cat | kubectl apply -n ${K8S_NAMESPACE} -f -" >> "${LOG_DIR}/stress.log" 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: ${stress_pod_name}
  labels:
    app: stage-b-stress
    run-tag: ${RUN_TAG}
    stress-profile: ${profile}
    stress-target: ${target}
spec:
  restartPolicy: Never
  nodeName: ${target}
  containers:
    - name: stress
      image: ${STAGE_B_STRESS_IMAGE}
      imagePullPolicy: IfNotPresent
      command:
        - /bin/sh
        - -lc
      args:
        - |-
          ${cmd}
EOF

  if ! run_remote kubectl wait -n "${K8S_NAMESPACE}" --for=condition=Ready pod/"${stress_pod_name}" --timeout="${STAGE_B_STRESS_READY_TIMEOUT_SECONDS}s" >> "${LOG_DIR}/stress.log" 2>&1; then
    echo "ERROR: stress pod ${stress_pod_name} did not become Ready" >> "${LOG_DIR}/stress.log"
    run_remote kubectl describe -n "${K8S_NAMESPACE}" pod "${stress_pod_name}" >> "${LOG_DIR}/stress.log" 2>&1 || true
    run_remote kubectl logs -n "${K8S_NAMESPACE}" "${stress_pod_name}" --tail=40 >> "${LOG_DIR}/stress.log" 2>&1 || true
    run_remote kubectl describe node "${target}" >> "${LOG_DIR}/stress.log" 2>&1 || true
    run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${stress_pod_name}" --ignore-not-found --wait=false >> "${LOG_DIR}/stress.log" 2>&1 || true
    return 1
  fi

  echo "${stress_pod_name}"
}

wait_for_stress_completion() {
  local stress_pod_name="$1"
  local wait_timeout_seconds=$(( STAGE_B_STRESS_TIMEOUT + 120 ))

  if ! run_remote kubectl wait -n "${K8S_NAMESPACE}" --for=jsonpath='{.status.phase}'=Succeeded pod/"${stress_pod_name}" --timeout="${wait_timeout_seconds}s" >> "${LOG_DIR}/stress.log" 2>&1; then
    echo "ERROR: stress pod ${stress_pod_name} did not complete successfully" >> "${LOG_DIR}/stress.log"
    run_remote kubectl describe -n "${K8S_NAMESPACE}" pod "${stress_pod_name}" >> "${LOG_DIR}/stress.log" 2>&1 || true
    run_remote kubectl logs -n "${K8S_NAMESPACE}" "${stress_pod_name}" --tail=40 >> "${LOG_DIR}/stress.log" 2>&1 || true
    run_remote kubectl describe node "${target}" >> "${LOG_DIR}/stress.log" 2>&1 || true
    run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${stress_pod_name}" --ignore-not-found --wait=false >> "${LOG_DIR}/stress.log" 2>&1 || true
    return 1
  fi

  run_remote kubectl logs -n "${K8S_NAMESPACE}" "${stress_pod_name}" --tail=20 >> "${LOG_DIR}/stress.log" 2>&1 || true
  run_remote kubectl delete -n "${K8S_NAMESPACE}" pod "${stress_pod_name}" --ignore-not-found --wait=false >> "${LOG_DIR}/stress.log" 2>&1 || true
}

collect_metrics() {
  local pod_name="$1"
  local output_csv="$2"
  local arm="$3"
  local arm_order="$4"
  local trial="$5"
  local phase_name="$6"
  local phase_kind="$7"
  local stress_profile="$8"
  local stress_target="$9"
  local decision_mode="${10}"
  local expected_node="${11}"
  local snapshot_path="${12}"
  local ranking_path="${13}"
  local workload_cpu_m="${14}"
  local workload_memory_mib="${15}"
  local decision_total_score="${16}"
  local decision_predicted_load="${17}"
  local decision_base_predicted_load="${18}"
  local decision_anomaly_risk="${19}"
  local decision_cpu_request_fraction="${20}"
  local decision_memory_request_fraction="${21}"
  local decision_capacity_penalty="${22}"
  local decision_prediction_source="${23}"

  "${PYTHON_BIN}" "${METRIC_SCRIPT}" \
    --pod "${pod_name}" \
    --output "${output_csv}" \
    --control-node "${CONTROL_NODE}" \
    --zone "${ZONE}" \
    --contention-relative-cpu-threshold "${STAGE_B_CONTENTION_REL_CPU_THRESHOLD}" \
    --anomaly-relative-cpu-threshold "${STAGE_B_ANOMALY_REL_CPU_THRESHOLD}" \
    --arm "${arm}" \
    --arm-order "${arm_order}" \
    --trial "${trial}" \
    --phase-name "${phase_name}" \
    --phase-kind "${phase_kind}" \
    --stress-profile "${stress_profile}" \
    --stress-target "${stress_target}" \
    --decision-mode "${decision_mode}" \
    --expected-node "${expected_node}" \
    --snapshot-path "${snapshot_path}" \
    --ranking-path "${ranking_path}" \
    --workload-cpu-m "${workload_cpu_m}" \
    --workload-memory-mib "${workload_memory_mib}" \
    --decision-total-score "${decision_total_score}" \
    --decision-predicted-load "${decision_predicted_load}" \
    --decision-base-predicted-load "${decision_base_predicted_load}" \
    --decision-anomaly-risk "${decision_anomaly_risk}" \
    --decision-cpu-request-fraction "${decision_cpu_request_fraction}" \
    --decision-memory-request-fraction "${decision_memory_request_fraction}" \
    --decision-capacity-penalty "${decision_capacity_penalty}" \
    --decision-prediction-source "${decision_prediction_source}" 2>&1 | tee -a "${LOG_DIR}/collect_metrics.log"
}

run_decision() {
  local arm="$1"
  local arm_order="$2"
  local trial="$3"
  local phase_name="$4"
  local phase_kind="$5"
  local stress_profile="$6"
  local stress_target="$7"
  local workload="$8"

  local pod_name
  pod_name="$(sanitize_k8s_name "${workload}-${arm}-${phase_name}-t${trial}")"
  local snapshot_file="${ARTIFACT_DIR}/${arm}/${pod_name}_snapshot.csv"
  local ranking_file=""
  local expected_node=""
  local decision_mode="${BASELINE_DECISION_MODE}"
  local output_csv="${BASELINE_CSV}"
  local scheduler_name=""
  local node_name=""
  local eligible_nodes=""
  local -a eligible_node_array=()
  local workload_cpu_m
  local workload_memory_mib
  local decision_total_score=""
  local decision_predicted_load=""
  local decision_base_predicted_load=""
  local decision_anomaly_risk=""
  local decision_cpu_request_fraction=""
  local decision_memory_request_fraction=""
  local decision_capacity_penalty=""
  local decision_prediction_source=""

  workload_cpu_m="$(workload_cpu_millicores "${workload}")"
  workload_memory_mib="$(workload_memory_mib "${workload}")"
  eligible_nodes="$(eligible_nodes_for_workload "${workload}")"
  read -r -a eligible_node_array <<< "${eligible_nodes}"

  if (( ${#eligible_node_array[@]} == 0 )); then
    echo "ERROR: no eligible nodes resolved for workload ${workload}" >&2
    exit 1
  fi

  if [[ "${arm}" == "custom" ]]; then
    seed_custom_history_if_needed
  fi

  capture_snapshot "${snapshot_file}" "${pod_name}_snapshot"

  if [[ "${arm}" == "baseline" ]]; then
    case "${BASELINE_DECISION_MODE}" in
      matched_round_robin_and_pin)
        expected_node="$(baseline_target_for "${trial}" "${phase_name}" "${workload}" "${eligible_nodes}")"
        node_name="${expected_node}"
        ;;
      default_scheduler)
        decision_mode="default_scheduler"
        scheduler_name="${DEFAULT_SCHEDULER_NAME}"
        expected_node=""
        node_name=""
        ;;
      *)
        echo "ERROR: unsupported BASELINE_DECISION_MODE=${BASELINE_DECISION_MODE}" >&2
        exit 1
        ;;
    esac
  elif [[ "${arm}" == "custom" ]]; then
    append_snapshot_to_history "${snapshot_file}"
    ranking_file="${ARTIFACT_DIR}/${arm}/${pod_name}_ranking.json"
    "${PYTHON_BIN}" "${RANK_SCRIPT}" \
      --input "${CUSTOM_HISTORY}" \
      --model-dir "${MODEL_DIR_PATH}" \
      --eligible-nodes "${eligible_node_array[@]}" \
      --window "${WINDOW}" \
      --pred-weight "${PRED_WEIGHT}" \
      --anomaly-weight "${ANOMALY_WEIGHT}" \
      --anomaly-history "${ANOMALY_HISTORY}" \
      --anomaly-source "${ANOMALY_SOURCE}" \
      --nsa-num-detectors "${NSA_NUM_DETECTORS}" \
      --nsa-radius "${NSA_RADIUS}" \
      --kmeans-threshold-std "${KMEANS_THRESHOLD_STD}" \
      --z-threshold "${Z_THRESHOLD}" \
      --adaptive-weighting "${ADAPTIVE_WEIGHTING}" \
      --adaptive-risk-low "${ADAPTIVE_RISK_LOW}" \
      --adaptive-risk-high "${ADAPTIVE_RISK_HIGH}" \
      --adaptive-max-shift "${ADAPTIVE_MAX_SHIFT}" \
      --adaptive-min-prediction-weight "${ADAPTIVE_MIN_PREDICTION_WEIGHT}" \
      --adaptive-max-prediction-weight "${ADAPTIVE_MAX_PREDICTION_WEIGHT}" \
      --contention-relative-cpu-threshold "${CONTENTION_RELATIVE_CPU_THRESHOLD}" \
      --contention-penalty-factor "${CONTENTION_PENALTY_FACTOR}" \
      --safe-relative-cpu-threshold "${SAFE_RELATIVE_CPU_THRESHOLD}" \
      --unsafe-penalty-factor "${UNSAFE_PENALTY_FACTOR}" \
      --avoid-unsafe-nodes "${AVOID_UNSAFE_NODES}" \
      --node-capacities "${CAPACITY_JSON}" \
      --workload-cpu-m "${workload_cpu_m}" \
      --workload-memory-mib "${workload_memory_mib}" \
      --output "${ranking_file}" 2>&1 | tee -a "${LOG_DIR}/rank_live.log"

    IFS=$'\t' read -r expected_node decision_total_score decision_predicted_load decision_base_predicted_load decision_anomaly_risk decision_cpu_request_fraction decision_memory_request_fraction decision_capacity_penalty decision_prediction_source < <(parse_ranking_fields "${ranking_file}")

    if [[ -z "${expected_node}" ]]; then
      echo "ERROR: rank engine did not return a best node for ${pod_name}" >&2
      exit 1
    fi

    decision_mode="rank_and_pin"
    output_csv="${CUSTOM_CSV}"
    scheduler_name=""
    node_name="${expected_node}"
  fi

  echo "Deploying ${pod_name} (arm=${arm}, phase=${phase_name}, workload=${workload}, target=${expected_node:-scheduler-driven}, eligible_nodes=${eligible_nodes})" | tee -a "${LOG_DIR}/stage_b.log"
  apply_workload "${workload}" "${pod_name}" "${scheduler_name}" "${node_name}" "${eligible_nodes}"
  collect_metrics \
    "${pod_name}" \
    "${output_csv}" \
    "${arm}" \
    "${arm_order}" \
    "${trial}" \
    "${phase_name}" \
    "${phase_kind}" \
    "${stress_profile}" \
    "${stress_target}" \
    "${decision_mode}" \
    "${expected_node}" \
    "${snapshot_file}" \
    "${ranking_file}" \
    "${workload_cpu_m}" \
    "${workload_memory_mib}" \
    "${decision_total_score}" \
    "${decision_predicted_load}" \
    "${decision_base_predicted_load}" \
    "${decision_anomaly_risk}" \
    "${decision_cpu_request_fraction}" \
    "${decision_memory_request_fraction}" \
    "${decision_capacity_penalty}" \
    "${decision_prediction_source}"
  cleanup_pod "${pod_name}"
  sleep "${STAGE_B_WASHOUT_SECONDS}"
}

run_phase() {
  local trial="$1"
  local phase_name="$2"
  local phase_kind="$3"
  local stress_profile="$4"
  local stress_target="$5"
  local arm_sequence
  local arm_position

  arm_sequence="$(arm_order_for_trial "${trial}")"
  echo "== Trial ${trial} | phase=${phase_name} | arm order=${arm_sequence} ==" | tee -a "${LOG_DIR}/stage_b.log"

  for workload in "${WORKLOAD_ARRAY[@]}"; do
    arm_position=0
    for arm in ${arm_sequence}; do
      arm_position=$((arm_position + 1))
      run_decision "${arm}" "${arm_position}" "${trial}" "${phase_name}" "${phase_kind}" "${stress_profile}" "${stress_target}" "${workload}"
    done
  done
}

echo "== Redesigned Stage B Authoritative Evaluation ==" | tee "${LOG_DIR}/stage_b.log"
echo "tag=${RUN_TAG}" | tee -a "${LOG_DIR}/stage_b.log"
echo "baseline decision mode=${BASELINE_DECISION_MODE}" | tee -a "${LOG_DIR}/stage_b.log"
echo "custom decision mode=rank_and_pin" | tee -a "${LOG_DIR}/stage_b.log"

export_node_capacities

for (( trial=1; trial<=STAGE_B_RUNS; trial++ )); do
  run_phase "${trial}" "normal" "normal" "" ""

  for idx in "${!STRESS_PROFILE_ARRAY[@]}"; do
    profile="${STRESS_PROFILE_ARRAY[$idx]}"
    target="$(pick_stress_target "${trial}" "$((idx + 1))")"
    echo "Starting stress profile=${profile} on ${target}" | tee -a "${LOG_DIR}/stage_b.log"
    stress_pod_name="$(start_stress "${profile}" "${target}")"
    sleep "${STAGE_B_STRESS_STARTUP_SECONDS}"
    run_phase "${trial}" "stress_${profile}" "stress" "${profile}" "${target}"
    wait_for_stress_completion "${stress_pod_name}"
  done
done

"${PYTHON_BIN}" "${ANALYSE_SCRIPT}" \
  --baseline "${BASELINE_CSV}" \
  --custom "${CUSTOM_CSV}" \
  --pi-node-name "${PI_NODE_NAME:-raspberrypi}" \
  --node-capacities "${CAPACITY_JSON}" \
  --output "${SUMMARY_JSON}" | tee -a "${LOG_DIR}/stage_b.log"

cat > "${RESULT_DIR}/stage_b_metadata.txt" <<EOF
run_tag=${RUN_TAG}
namespace=${K8S_NAMESPACE}
baseline_decision_mode=${BASELINE_DECISION_MODE}
default_scheduler_name=${DEFAULT_SCHEDULER_NAME}
custom_scheduler_name=${CUSTOM_SCHEDULER_NAME}
custom_decision_mode=rank_and_pin
control_node=${CONTROL_NODE}
zone=${ZONE}
worker_nodes=${WORKER_NODES}
nodes=${NODES}
runs_per_phase=${STAGE_B_RUNS}
workloads=${STAGE_B_WORKLOADS}
pi_node_name=${PI_NODE_NAME}
pi_eligible_workloads=${PI_ELIGIBLE_WORKLOADS}
arm_order_mode=${STAGE_B_ARM_ORDER_MODE}
warmup_samples=${STAGE_B_WARMUP_SAMPLES}
warmup_interval_s=${STAGE_B_WARMUP_INTERVAL}
washout_seconds=${STAGE_B_WASHOUT_SECONDS}
stress_profiles=${STAGE_B_STRESS_PROFILES}
stress_targets=${STAGE_B_STRESS_TARGETS}
stress_timeout=${STAGE_B_STRESS_TIMEOUT}
stress_startup_seconds=${STAGE_B_STRESS_STARTUP_SECONDS}
stress_image=${STAGE_B_STRESS_IMAGE}
stress_ready_timeout_seconds=${STAGE_B_STRESS_READY_TIMEOUT_SECONDS}
stress_cpu_workers=${STAGE_B_STRESS_CPU_WORKERS}
stress_reserved_cpus=${STAGE_B_STRESS_RESERVED_CPUS}
stress_memory_mib=${STAGE_B_STRESS_MEMORY_MIB}
stress_mixed_memory_mib=${STAGE_B_STRESS_MIXED_MEMORY_MIB}
stress_reserved_memory_mib=${STAGE_B_STRESS_RESERVED_MEMORY_MIB}
stress_min_memory_mib=${STAGE_B_STRESS_MIN_MEMORY_MIB}
stress_nice_level=${STAGE_B_STRESS_NICE_LEVEL}
snapshot_max_attempts=${SNAPSHOT_MAX_ATTEMPTS}
snapshot_retry_delay_seconds=${SNAPSHOT_RETRY_DELAY_SECONDS}
model_dir=${MODEL_DIR_PATH}
window=${WINDOW}
pred_weight=${PRED_WEIGHT}
anomaly_weight=${ANOMALY_WEIGHT}
anomaly_history=${ANOMALY_HISTORY}
anomaly_source=${ANOMALY_SOURCE}
nsa_num_detectors=${NSA_NUM_DETECTORS}
nsa_radius=${NSA_RADIUS}
kmeans_threshold_std=${KMEANS_THRESHOLD_STD}
z_threshold=${Z_THRESHOLD}
adaptive_weighting=${ADAPTIVE_WEIGHTING}
adaptive_risk_low=${ADAPTIVE_RISK_LOW}
adaptive_risk_high=${ADAPTIVE_RISK_HIGH}
adaptive_max_shift=${ADAPTIVE_MAX_SHIFT}
adaptive_min_prediction_weight=${ADAPTIVE_MIN_PREDICTION_WEIGHT}
adaptive_max_prediction_weight=${ADAPTIVE_MAX_PREDICTION_WEIGHT}
contention_relative_cpu_threshold=${CONTENTION_RELATIVE_CPU_THRESHOLD}
contention_penalty_factor=${CONTENTION_PENALTY_FACTOR}
safe_relative_cpu_threshold=${SAFE_RELATIVE_CPU_THRESHOLD}
unsafe_penalty_factor=${UNSAFE_PENALTY_FACTOR}
avoid_unsafe_nodes=${AVOID_UNSAFE_NODES}
contention_rel_cpu_threshold=${STAGE_B_CONTENTION_REL_CPU_THRESHOLD}
anomaly_rel_cpu_threshold=${STAGE_B_ANOMALY_REL_CPU_THRESHOLD}
completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "Stage B complete." | tee -a "${LOG_DIR}/stage_b.log"
echo "Summary: ${SUMMARY_JSON}" | tee -a "${LOG_DIR}/stage_b.log"