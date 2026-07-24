#!/bin/bash
# TensorStore + GCSFuse Workload Launcher
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "=================================================="
echo " Starting TensorStore + GCSFuse Workload Launcher "
echo "=================================================="

# Install Python requirements
STEP_START=$(date +%s)
echo "Installing Python dependencies..."
pip3 install --no-cache-dir -r /workload/configs/requirements.txt

if [[ -n "${REQUIREMENTS:-}" ]] && [[ "${REQUIREMENTS:-}" != "none" ]]; then
  pip3 install $REQUIREMENTS
fi
echo "[BENCHMARK] Python dependencies setup finished in $(( $(date +%s) - STEP_START )) seconds."

# Determine mount path
MOUNT_PATH="${CKPT_WRITE_PATH:-/gcs/checkpoints}"
if [[ "$MOUNT_PATH" == gs://* ]]; then
  echo "Warning: CKPT_WRITE_PATH is $MOUNT_PATH, defaulting to /gcs/checkpoints for GCSFuse mount."
  MOUNT_PATH="/gcs/checkpoints"
fi

echo "Running TensorStore + GCSFuse benchmark against mount path: $MOUNT_PATH"

SHAPE="${TENSORSTORE_SHAPE:-1000,1000,100}"
CHUNKS="${TENSORSTORE_CHUNKS:-100,100,100}"
DTYPE="${TENSORSTORE_DTYPE:-float32}"
DRIVER="${TENSORSTORE_DRIVER:-zarr}"
ITERATIONS="${TENSORSTORE_ITERATIONS:-1}"

WORKERS="${NUM_WORKERS:-1}"
NODE_RANK="${JOB_COMPLETION_INDEX:-${NODE_RANK:-0}}"
NUM_NODES="${NNODES:-${NODES:-1}}"
PER_WORKER_SHAPE="${TENSORSTORE_PER_WORKER_SHAPE:-false}"

EXTRA_ARGS=()
if [[ "$PER_WORKER_SHAPE" == "true" ]] || [[ "$PER_WORKER_SHAPE" == "1" ]]; then
  EXTRA_ARGS+=("--per-worker-shape")
fi

echo "Node Rank: $NODE_RANK / $NUM_NODES | Local workers per node: $WORKERS"

python3 -u /workload/tensorstore_bench.py \
  --mount-path "$MOUNT_PATH" \
  --shape "$SHAPE" \
  --chunks "$CHUNKS" \
  --dtype "$DTYPE" \
  --driver "$DRIVER" \
  --iterations "$ITERATIONS" \
  --num-workers "$WORKERS" \
  --node-rank "$NODE_RANK" \
  --num-nodes "$NUM_NODES" \
  "${EXTRA_ARGS[@]}" \
  --verify

echo "TensorStore + GCSFuse benchmark run completed on node $NODE_RANK."
