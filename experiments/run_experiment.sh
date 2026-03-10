#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ZONE="europe-west2-c"

CONTROL="k3s-control"
WORKER="k3s-worker-2"

RAW_DATA="$PROJECT_DIR/dataset/dataset.csv"
LABELLED_DATA="$PROJECT_DIR/dataset/labelled.csv"
PROCESSED_DATA="$PROJECT_DIR/dataset/processed.csv"
PHASES_FILE="$PROJECT_DIR/dataset/phases.json"
RESULTS_DIR="$PROJECT_DIR/experiments/results"

# ── Phase 1: Data Collection ──

echo "Starting collector..."
python "$PROJECT_DIR/data_collection/netdata_collector.py" &
COLLECTOR_PID=$!

# record phase timestamps for labelling
echo "[" > "$PHASES_FILE"

BASELINE_START=$(date +%s)
echo "Baseline phase (2 minutes)"
sleep 120
BASELINE_END=$(date +%s)
echo "  {\"type\": \"baseline\", \"start\": $BASELINE_START, \"end\": $BASELINE_END}," >> "$PHASES_FILE"

CPU_START=$(date +%s)
echo "CPU stress on worker node"
gcloud compute ssh $WORKER --zone=$ZONE --command="stress --cpu 2 --timeout 120"
CPU_END=$(date +%s)
echo "  {\"type\": \"stress\", \"name\": \"cpu\", \"target\": \"$WORKER\", \"start\": $CPU_START, \"end\": $CPU_END}," >> "$PHASES_FILE"

MEM_START=$(date +%s)
echo "Memory stress on control node"
gcloud compute ssh $CONTROL --zone=$ZONE --command="stress --vm 1 --vm-bytes 1G --timeout 120"
MEM_END=$(date +%s)
echo "  {\"type\": \"stress\", \"name\": \"memory\", \"target\": \"$CONTROL\", \"start\": $MEM_START, \"end\": $MEM_END}," >> "$PHASES_FILE"

IO_START=$(date +%s)
echo "IO stress on worker node"
gcloud compute ssh $WORKER --zone=$ZONE --command="stress --io 4 --timeout 120"
IO_END=$(date +%s)
echo "  {\"type\": \"stress\", \"name\": \"io\", \"target\": \"$WORKER\", \"start\": $IO_START, \"end\": $IO_END}," >> "$PHASES_FILE"

RECOVERY_START=$(date +%s)
echo "Recovery phase (2 minutes)"
sleep 120
RECOVERY_END=$(date +%s)
echo "  {\"type\": \"recovery\", \"start\": $RECOVERY_START, \"end\": $RECOVERY_END}" >> "$PHASES_FILE"

echo "]" >> "$PHASES_FILE"

echo "Stopping collector"
kill $COLLECTOR_PID
sleep 2

echo "Phase timestamps saved to $PHASES_FILE"

# ── Phase 2: Labelling ──

echo ""
echo "Labelling data (per-node)..."
python "$PROJECT_DIR/preprocessing/label_data.py" \
    --input "$RAW_DATA" \
    --output "$LABELLED_DATA" \
    --phases "$PHASES_FILE"

# ── Phase 3 & 4: Preprocessing + Detection for each window size ──

for WINDOW in 3 5 10; do
    echo ""
    echo "========================================"
    echo "  Window size: $WINDOW (worker-only)"
    echo "========================================"

    EXPERIMENT_DIR="$RESULTS_DIR/window_${WINDOW}"
    PROCESSED="$PROJECT_DIR/dataset/processed_w${WINDOW}.csv"

    echo "Preprocessing (window=$WINDOW, node=$WORKER)..."
    python "$PROJECT_DIR/preprocessing/clean_metrics.py" \
        --input "$LABELLED_DATA" \
        --output "$PROCESSED" \
        --window "$WINDOW" \
        --node "$WORKER"

    echo "Running K-Means detection..."
    python "$PROJECT_DIR/experiments/detection/kmeans_detector.py" \
        --input "$PROCESSED" \
        --output "$EXPERIMENT_DIR/kmeans_results.json"

    echo "Running NSA detection..."
    python "$PROJECT_DIR/experiments/detection/nsa_detector.py" \
        --input "$PROCESSED" \
        --output "$EXPERIMENT_DIR/nsa_results.json"
done

echo ""
echo "Experiment complete. Results saved to $RESULTS_DIR/"