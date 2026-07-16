#!/bin/bash

# CPU emulator launcher. Each pod on c4-standard-192
# runs this; torchrun then forks GPUS_PER_NODE worker processes per pod (4 by
# default), so 2 nodes x 4 = 8 ranks total. Per-node ranks are capped at 4 (not
# 8) so a checkpoint-restoring run fits the 720GB c4-standard-192 host RAM; see
# values_base.yaml.

set -euo pipefail

export PYTHONUNBUFFERED=1

# The default workload image is nvcr.io/nvidia/pytorch:26.05-py3 (see
# values_base.yaml), which already ships curl/ca-certificates, so this guard is
# a no-op there. It exists for minimal Debian-based fallback images (e.g.
# python:3.11-slim) that omit curl/ca-certificates, which the gcloud install +
# model download below both need. Install once per pod; subsequent pip steps
# fail clearly if this step fails.
if ! command -v curl >/dev/null 2>&1; then
  echo "Installing curl + ca-certificates (needed for gcloud download)..."
  apt-get update -qq
  apt-get install -y --no-install-recommends curl ca-certificates
  rm -rf /var/lib/apt/lists/*
fi

STEP_START=$(date +%s)
echo "Installing standalone gcloud CLI..."
cd /tmp
curl -sSO https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
rm google-cloud-cli-linux-x86_64.tar.gz
export PATH=$PATH:/tmp/google-cloud-sdk/bin
cd -
echo "[BENCHMARK] gcloud CLI installation finished in $(( $(date +%s) - STEP_START )) seconds."

# If MODEL_ID is a GCS path, pull the weights once per pod. cpu_sim.py will
# then load from /tmp/<basename> with local_files_only=True, so the ranks
# on this node do not race on the HuggingFace API. Skipping the download if
# the directory already exists keeps pod restarts cheap.
if [[ "${MODEL_ID:-}" == gs://* ]]; then
  echo "MODEL_ID is a GCS path: $MODEL_ID"
  DIR_NAME=$(basename "${MODEL_ID%/}")
  LOCAL_MODEL_PATH="/tmp/$DIR_NAME"

  if [[ ! -d "$LOCAL_MODEL_PATH" ]]; then
    STEP_START=$(date +%s)
    echo "Downloading model from GCS to $LOCAL_MODEL_PATH..."
    /tmp/google-cloud-sdk/bin/gcloud storage cp -r "${MODEL_ID%/}" /tmp/
    echo "[BENCHMARK] Model download finished in $(( $(date +%s) - STEP_START )) seconds."
  else
    echo "Model already exists at $LOCAL_MODEL_PATH, skipping download."
  fi
fi

STEP_START=$(date +%s)
pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
pip3 install --no-cache-dir -r /workload/configs/requirements.txt

if [[ -n "${REQUIREMENTS:-}" ]]; then
  pip3 install $REQUIREMENTS
fi
echo "[BENCHMARK] Python dependencies setup finished in $(( $(date +%s) - STEP_START )) seconds."

# JOB_COMPLETION_INDEX is set by the K8s Indexed Job (one value per pod,
# 0..NNODES-1). torchrun consumes it as --node_rank.
export NODE_RANK=$JOB_COMPLETION_INDEX
export HYDRA_FULL_ERROR=1

echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"

# Gloo (the CPU collective backend used by DDPStrategy in cpu_sim.py) does not
# auto-discover the right NIC across pods reliably; pin it to the pod's
# primary interface. With hostNetwork: false this is always eth0 inside the
# pod regardless of the c4 host's underlying NIC name (ens4/etc.).
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TOKENIZERS_PARALLELISM=false

# gRPC DirectPath / HTTP2 keepalive and idle timeout configurations
export GRPC_ARG_KEEPALIVE_TIME_MS=${GRPC_ARG_KEEPALIVE_TIME_MS:-30000}
export GRPC_ARG_KEEPALIVE_TIMEOUT_MS=${GRPC_ARG_KEEPALIVE_TIMEOUT_MS:-20000}
export GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS=${GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS:-1}
export GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA=${GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA:-0}
export GRPC_ARG_CLIENT_IDLE_TIMEOUT_MS=${GRPC_ARG_CLIENT_IDLE_TIMEOUT_MS:-3600000}

# Parallel training strategy: ddp (default), fsdp_sharded, or fsdp_full.
export TRAINING_STRATEGY=${TRAINING_STRATEGY:-ddp}

# Training parameters.
export NUM_TRAIN_EPOCHS=1
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-8}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}

# Enable Python fault handler so a segfault in any of the 8 ranks dumps
# a stack trace into pod logs.
export PYTHONFAULTHANDLER=1
# DataLoader workers do their own tokenization; cap BLAS threads per worker
# so 4 ranks * 16 workers stay within the 192 vCPUs on c4-standard-192.
# (Lower DATALOADER_NUM_WORKERS if step-time IO timing looks CPU-bound:
# 4 ranks * 16 workers = 64 procs vs 192 vCPUs.)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONPATH=${PYTHONPATH:-}:/workload/configs

torchrun \
  --nproc_per_node="${GPUS_PER_NODE:-4}" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$PYTHON_MAIN"
