#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

WORKLOADS_DIR="$SCRIPT_DIR/workloads"
EVAL_SCRIPT="$SCRIPT_DIR/evaluation/collect_scheduling_metrics.py"
RESULTS_DIR="$SCRIPT_DIR/evaluation/results"
OUTPUT_CSV="$RESULTS_DIR/baseline_scheduling.csv"
NETDATA_COLLECTOR="$PROJECT_DIR/data_collection/netdata_collector.py"
RESOURCE_CSV="$RESULTS_DIR/resource_utilisation.csv"

ZONE="europe-west2-c"
CONTROL="k3s-control"
WORKER="k3s-worker-2"
WORKER2="k3s-worker-3"

NUM_RUNS=5

# helper: run kubectl on the control node
remote_kubectl() {
    gcloud compute ssh "$CONTROL" --zone="$ZONE" --command="$*"
}

mkdir -p "$RESULTS_DIR"

# remove previous results
rm -f "$OUTPUT_CSV"

echo "========================================"
echo "  Baseline Scheduler Evaluation"
echo "========================================"

# ── Experiment 1: Normal conditions ──

echo ""
echo "── Experiment 1: Normal conditions ──"
echo ""

for i in $(seq 1 $NUM_RUNS); do
    for WORKLOAD in cpu-pod memory-pod mixed-pod; do
        POD_NAME="${WORKLOAD}-normal-${i}"
        echo "Run $i: Deploying $WORKLOAD..."

        # create pod with unique name via remote kubectl
        sed "s/name: .*/name: ${POD_NAME}/" "$WORKLOADS_DIR/${WORKLOAD}.yaml" | \
            gcloud compute ssh "$CONTROL" --zone="$ZONE" --command="cat | kubectl apply -f -"

        # collect scheduling metrics
        python "$EVAL_SCRIPT" --pod "$POD_NAME" --output "$OUTPUT_CSV" \
            --control-node "$CONTROL" --zone "$ZONE"

        # wait for pod to finish then clean up
        remote_kubectl kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=60s 2>/dev/null
        sleep 5
        remote_kubectl kubectl delete pod "$POD_NAME" --ignore-not-found --wait=false
        sleep 2
    done
done

# ── Experiment 2: Under stress (worker-3 stressed) ──

echo ""
echo "── Experiment 2: Worker-3 under CPU stress ──"
echo ""

echo "Starting stress on $WORKER2..."
gcloud compute ssh "$WORKER2" --zone="$ZONE" --command="stress --cpu 2 --timeout 180" &
STRESS_PID=$!
sleep 10  # let stress ramp up

for i in $(seq 1 $NUM_RUNS); do
    for WORKLOAD in cpu-pod memory-pod mixed-pod; do
        POD_NAME="${WORKLOAD}-stress-${i}"
        echo "Run $i: Deploying $WORKLOAD under stress..."

        sed "s/name: .*/name: ${POD_NAME}/" "$WORKLOADS_DIR/${WORKLOAD}.yaml" | \
            gcloud compute ssh "$CONTROL" --zone="$ZONE" --command="cat | kubectl apply -f -"

        python "$EVAL_SCRIPT" --pod "$POD_NAME" --output "$OUTPUT_CSV" \
            --control-node "$CONTROL" --zone "$ZONE"

        remote_kubectl kubectl wait --for=condition=Ready pod/"$POD_NAME" --timeout=60s 2>/dev/null
        sleep 5
        remote_kubectl kubectl delete pod "$POD_NAME" --ignore-not-found --wait=false
        sleep 2
    done
done

echo "Waiting for stress to finish..."
wait $STRESS_PID 2>/dev/null

# ── Analysis ──

echo ""
echo "Analysing results..."
python "$SCRIPT_DIR/evaluation/analyse_results.py" \
    --input "$OUTPUT_CSV" \
    --output "$RESULTS_DIR/baseline_analysis.json"

echo ""
echo "========================================"
echo "  Evaluation complete"
echo "  Results: $RESULTS_DIR/"
echo "========================================"
