# Locked Evaluation Policy: Default vs Custom Scheduler

## Primary objective
Use the locked utility objective already validated in offline policy selection:

- objective: utility
- safe_placement_rate: 0.45
- anomalous_rate: 0.25
- high_contention_decision_rate: 0.15
- latency: 0.10
- placement_fairness_percent: 0.05

Source artifact:
- scheduler-prediction/custom-scheduler/results/offline_policy_eval_anomsrc_nsa.json

## Locked custom policy parameters
Use the NSA custom scheduler configuration selected on locked splits:

- anomaly_source: nsa
- pred_weight: 0.9
- anomaly_weight: 0.1
- z_threshold: 2.5
- anomaly_history: 45

## Comparison policy for online experiments

### Stage A (current, already runnable)
Telemetry-grounded comparison:
- Default scheduler baseline run.
- Rank-engine smoke + campaign telemetry.
- Compare startup/latency distributions and anomaly exposure metrics.

This stage is useful for operational validation and trend comparison.

### Stage B (authoritative scheduler comparison, required for strongest claim)
To claim "custom scheduler outperforms default scheduler" at scheduler-policy level, enforce scheduler authority:

1. Default arm:
- pods scheduled with schedulerName: default-scheduler

2. Custom arm:
- pods scheduled by a custom scheduler path (kube-scheduler plugin/profile) OR an equivalent authoritative binding path before default scheduling.

3. Pairing rule:
- Matched workload sequence and timing across arms.
- Same stress/control scenarios.
- Same run count and random seed.

4. Reporting rule:
- Report utility components separately (safety, anomaly, contention, latency, fairness).
- Report final utility score using locked objective weights above.

## Decision
Use Stage A for immediate execution and diagnostics.
Use Stage B for final dissertation-grade scheduler comparison claim.

## Why Stage B is required for final claim
Without scheduler authority in the custom arm, results can be influenced by default scheduler decisions. That weakens causal attribution of observed gains to the custom policy itself.
