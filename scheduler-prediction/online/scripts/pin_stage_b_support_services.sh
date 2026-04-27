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
CUSTOM_SCHEDULER_HOSTNAME="${STAGE_B_SUPPORT_NODE_HOSTNAME:-${CONTROL_NODE}}"
NETDATA_STORAGE_MODE="${STAGE_B_SUPPORT_STORAGE_MODE:-pvc}"
NETDATA_STATEFUL_HOSTNAME="${STAGE_B_NETDATA_STATEFUL_HOSTNAME:-k3s-worker-2}"
NETDATA_SUPPORT_HOSTNAME="${STAGE_B_NETDATA_SUPPORT_HOSTNAME:-${CONTROL_NODE}}"
CUSTOM_SCHEDULER_DEPLOY="${CUSTOM_SCHEDULER_NAME:-custom-rank-scheduler}"

run_kubectl() {
  gcloud compute ssh "${CONTROL_NODE}" --zone="${ZONE}" --command="kubectl $*"
}

tolerations_json_for_host() {
  local hostname="$1"

  if [[ "${hostname}" == "${CONTROL_NODE}" ]]; then
    echo '[{"key":"node-role.kubernetes.io/control-plane","operator":"Exists","effect":"NoSchedule"}]'
    return
  fi

  echo '[]'
}

netdata_parent_patch_payload() {
  local hostname="$1"
  local storage_mode="$2"
  local tolerations_json="$3"

  if [[ "${storage_mode}" == "ephemeral" ]]; then
    cat <<EOF
{
  "spec": {
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/hostname": "${hostname}"
        },
        "tolerations": ${tolerations_json},
        "volumes": [
          {
            "name": "os-release",
            "hostPath": {
              "path": "/etc/os-release",
              "type": ""
            }
          },
          {
            "name": "configmap",
            "configMap": {
              "name": "netdata-conf-parent",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "configsecret",
            "secret": {
              "secretName": "netdata-conf-parent",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "database",
            "emptyDir": {}
          },
          {
            "name": "alarms",
            "emptyDir": {}
          }
        ]
      }
    }
  }
}
EOF
    return
  fi

  cat <<EOF
{
  "spec": {
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/hostname": "${hostname}"
        },
        "tolerations": ${tolerations_json},
        "volumes": [
          {
            "name": "os-release",
            "hostPath": {
              "path": "/etc/os-release",
              "type": ""
            }
          },
          {
            "name": "configmap",
            "configMap": {
              "name": "netdata-conf-parent",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "configsecret",
            "secret": {
              "secretName": "netdata-conf-parent",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "database",
            "persistentVolumeClaim": {
              "claimName": "netdata-parent-database"
            }
          },
          {
            "name": "alarms",
            "persistentVolumeClaim": {
              "claimName": "netdata-parent-alarms"
            }
          }
        ]
      }
    }
  }
}
EOF
}

netdata_k8s_state_patch_payload() {
  local hostname="$1"
  local storage_mode="$2"
  local tolerations_json="$3"

  if [[ "${storage_mode}" == "ephemeral" ]]; then
    cat <<EOF
{
  "spec": {
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/hostname": "${hostname}"
        },
        "tolerations": ${tolerations_json},
        "volumes": [
          {
            "name": "os-release",
            "hostPath": {
              "path": "/etc/os-release",
              "type": ""
            }
          },
          {
            "name": "configmap",
            "configMap": {
              "name": "netdata-conf-k8s-state",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "configsecret",
            "secret": {
              "secretName": "netdata-conf-k8s-state",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "varlib",
            "emptyDir": {}
          }
        ]
      }
    }
  }
}
EOF
    return
  fi

  cat <<EOF
{
  "spec": {
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/hostname": "${hostname}"
        },
        "tolerations": ${tolerations_json},
        "volumes": [
          {
            "name": "os-release",
            "hostPath": {
              "path": "/etc/os-release",
              "type": ""
            }
          },
          {
            "name": "configmap",
            "configMap": {
              "name": "netdata-conf-k8s-state",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "configsecret",
            "secret": {
              "secretName": "netdata-conf-k8s-state",
              "optional": true,
              "defaultMode": 420
            }
          },
          {
            "name": "varlib",
            "persistentVolumeClaim": {
              "claimName": "netdata-k8s-state-varlib"
            }
          }
        ]
      }
    }
  }
}
EOF
}

patch_deployment() {
  local deployment_name="$1"
  local payload="$2"

  run_kubectl patch deployment -n netdata "${deployment_name}" --type merge -p "'${payload}'"
}

case "${NETDATA_STORAGE_MODE}" in
  pvc)
    NETDATA_HOSTNAME="${NETDATA_STATEFUL_HOSTNAME}"
    ;;
  ephemeral)
    NETDATA_HOSTNAME="${NETDATA_SUPPORT_HOSTNAME}"
    ;;
  *)
    echo "ERROR: unsupported STAGE_B_SUPPORT_STORAGE_MODE=${NETDATA_STORAGE_MODE}" >&2
    exit 1
    ;;
esac

NETDATA_TOLERATIONS_JSON="$(tolerations_json_for_host "${NETDATA_HOSTNAME}")"
NETDATA_PARENT_PATCH_PAYLOAD="$(netdata_parent_patch_payload "${NETDATA_HOSTNAME}" "${NETDATA_STORAGE_MODE}" "${NETDATA_TOLERATIONS_JSON}")"
NETDATA_K8S_STATE_PATCH_PAYLOAD="$(netdata_k8s_state_patch_payload "${NETDATA_HOSTNAME}" "${NETDATA_STORAGE_MODE}" "${NETDATA_TOLERATIONS_JSON}")"

echo "== Pin Stage B Support Services =="
echo "control node: ${CONTROL_NODE} (${ZONE})"
echo "custom scheduler hostname: ${CUSTOM_SCHEDULER_HOSTNAME}"
echo "netdata storage mode: ${NETDATA_STORAGE_MODE}"
echo "netdata support hostname: ${NETDATA_HOSTNAME}"

echo "[1/4] Applying custom scheduler manifest"
bash "${SCRIPT_DIR}/deploy_custom_rank_scheduler.sh" "${ENV_FILE}"

echo "[2/4] Patching netdata parent deployment"
patch_deployment netdata-parent "${NETDATA_PARENT_PATCH_PAYLOAD}"

echo "[3/4] Patching netdata k8s-state deployment"
patch_deployment netdata-k8s-state "${NETDATA_K8S_STATE_PATCH_PAYLOAD}"

echo "[4/4] Waiting for rollouts and showing placement"
run_kubectl rollout status deployment/netdata-parent -n netdata --timeout=300s
run_kubectl rollout status deployment/netdata-k8s-state -n netdata --timeout=300s
run_kubectl rollout status deployment/"${CUSTOM_SCHEDULER_DEPLOY}" -n kube-system --timeout=300s
run_kubectl get pods -n netdata -o wide
run_kubectl get pods -n kube-system -l app="${CUSTOM_SCHEDULER_DEPLOY}" -o wide

echo "Support services aligned for Stage B."