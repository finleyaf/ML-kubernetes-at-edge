# Online Hybrid-vs-Default Test Workspace

This directory is for implementation and execution of online scheduler tests.
It keeps online operations separate from offline model development and report assets.

## Directory layout

- `config/online_test.env.example`: environment template for cluster/test settings.
- `scripts/preflight_cluster.sh`: validates local tools, SSH access, node prerequisites, and netdata endpoint.
- `scripts/run_online_campaign.sh`: generates/executes an online telemetry campaign plan.
- `scripts/run_readiness_smoke.sh`: verifies hybrid runtime ranking can execute on latest collected data.
- `results/`: logs and artifacts from online test runs.

## Quick start

1. Create env file:

   `cp scheduler-prediction/online/config/online_test.env.example scheduler-prediction/online/config/online_test.env`

2. Edit values in `online_test.env` for your cluster.

3. Run preflight checks:

   `bash scheduler-prediction/online/scripts/preflight_cluster.sh scheduler-prediction/online/config/online_test.env`

4. Run readiness smoke test (local model + latest run data):

   `bash scheduler-prediction/online/scripts/run_readiness_smoke.sh scheduler-prediction/online/config/online_test.env`

5. Run online campaign collection:

   `bash scheduler-prediction/online/scripts/run_online_campaign.sh scheduler-prediction/online/config/online_test.env`

## Stage B authoritative A/B (default vs custom scheduler)

1. Deploy custom scheduler profile:

   `bash scheduler-prediction/online/scripts/deploy_custom_rank_scheduler.sh scheduler-prediction/online/config/online_test.env`

2. Run matched-arm experiment:

   `bash scheduler-prediction/online/scripts/run_stage_b_matched_arms.sh scheduler-prediction/online/config/online_test.env`

3. Review outputs:

   - `scheduler-prediction/online/results/<tag>/baseline_arm_scheduling.csv`
   - `scheduler-prediction/online/results/<tag>/custom_arm_scheduling.csv`
   - `scheduler-prediction/online/results/<tag>/stage_b_summary.json`

Stage B now reports full locked utility components in `stage_b_summary.json`:

- `safe_placement_rate`
- `anomalous_rate`
- `high_contention_decision_rate`
- `avg_decision_latency_ms` (derived from scheduling latency)
- `placement_fairness_percent`
- per-arm `utility_score` and custom-minus-baseline `utility_delta`

Related env knobs in `online_test.env`:

- `STAGE_B_CONTENTION_REL_CPU_THRESHOLD`
- `STAGE_B_ANOMALY_REL_CPU_THRESHOLD`

## Cluster machine setup checklist

Run these setup actions on cluster machines before online tests:

1. On local operator machine:
   - `gcloud` installed and authenticated.
   - Access to target project/zone.
   - Python environment available for this repository.

2. On control node and workers:
   - `stress` installed (`sudo apt-get update && sudo apt-get install -y stress`).
   - SSH reachable from local machine via `gcloud compute ssh`.
   - Clock sync healthy (NTP enabled).

3. Telemetry stack:
   - Netdata agent/process reachable at expected endpoint from local machine.
   - Endpoint configured in env file (`NETDATA_BASE_URL`, default `http://localhost:20000`).

4. Kubernetes access:
   - Control node can run `kubectl` against the target cluster.
   - Namespace/workloads used for baseline comparison are available.

5. Python dependencies:
   - Install dependencies from:
     - `anomaly-detection/online-telemetry/requirements.txt`
     - `scheduler-prediction` dependencies in your active environment.

## Notes

- These scripts do not modify production scheduler settings.
- They prepare and validate the environment so you can safely run online comparisons against the default scheduler.
- Keep each test run under a unique tag so artifacts are isolated under `results/<tag>/`.
- The custom scheduler in this repository is a ranking engine (`custom-scheduler/rank_live.py`), not a deployed Kubernetes scheduler plugin.
- To enforce its selected node in-cluster, consume `best_node` from ranking output and submit workloads with `spec.nodeName` (or strong node affinity) targeting that node.
- Stage B scripts use explicit `spec.schedulerName` for both arms to preserve scheduler authority and causal attribution.
