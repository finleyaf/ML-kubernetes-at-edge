# ML Kubernetes at Edge

Anomaly-Aware Lightweight ML for Efficient Kubernetes Scheduling at the Edge.

This repository contains the complete implementation and evaluation artifacts for a  project that develops a lightweight Kubernetes scheduling pipeline for edge environments.

**[Read the full Dissertation Report here](./report/)**

## System Architecture

### High-Level Decision Flow
![System Design](./system-design.png)

### Cluster Implementation & Orchestration
![System Architecture](./system-architecture.jpg)

## Project summary

- Designed a hybrid scheduler that combines:
  - NSA-inspired anomaly monitoring on live Netdata telemetry.
  - Short horizon node load prediction.
  - Workload aware node capacity scoring.
- Implemented a complete development workflow from offline replication and model validation through to live matched-arm scheduler comparison.
- Evaluated the system in a controlled edge setting and showed that the custom scheduler:
  - Increased the safe placement rate from 64.17% to 75.83%.
  - Reduced anomalous placements from 35.83% to 24.17%.
  - Preserved mean scheduling latency, and improved startup and total execution time by approximately 23%.

## Key contributions

- Lightweight anomaly detection replication using synthetic and real Netdata telemetry.
- Short-horizon forecasting models for node-load prediction with bias-aware feature scaling.
- Hybrid ranking and binding logic in `scheduler-prediction/custom-scheduler/` that enforces safety-first placement while keeping runtime overhead low.
- Offline replay audit and locked policy selection to ensure claim-bearing results transfer to online deployment.
- Live K3s experiment orchestration with traceable decision artifacts, scheduling metrics, and fair matched-arm comparison.

## Repository structure

- `anomaly-detection/offline-replication/` — synthetic data replication, anomaly detector comparison, and evaluation.
- `anomaly-detection/online-telemetry/` — Netdata metric collection, telemetry preprocessing, and cluster dataset generation.
- `scheduler-prediction/prediction/` — predictor training, validation, and model selection.
- `scheduler-prediction/custom-scheduler/` — hybrid scheduler implementation, offline policy evaluation, and ranking tools.
- `scheduler-prediction/online/` — live experiment runner, cluster configuration, scripts, and result artifacts.
- `scheduler-prediction/baseline/` — baseline workload manifests and evaluation helpers used for comparison.
- `report/` — dissertation report source with full methodology, implementation details, and results discussion.

## Usage notes

Each major subsystem has its own dependencies and experiment-specific scripts. This top-level README is the primary entry point for understanding the project structure and core results.

For detailed reproduction or inspection, explore the relevant directories directly and consult any local README files where available.

## Technologies

- Python
- Kubernetes (K3s)
- Netdata telemetry
- scikit-learn, pandas, numpy
- Bash orchestration and experiment automation

## Outcome

This project demonstrates that a lightweight, anomaly-aware scheduler can outperform the default Kubernetes scheduler on placement safety and application startup metrics in a controlled edge-style deployment, while preserving low scheduling overhead and maintaining a traceable experimental workflow.

