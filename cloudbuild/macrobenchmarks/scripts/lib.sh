#!/usr/bin/env bash
# Shared helpers for the macrobenchmarks run-pipeline step scripts.
#
# Each step of ../macrobenchmarks-cloudbuild.yaml invokes one script in this
# directory. Cloud Build substitutions reach the scripts as environment
# variables (wired through each step's `env:` block) because Cloud Build does
# not substitute ${...} inside a file read from disk -- so the scripts read
# e.g. ${_ZONE} and ${PROJECT_ID} as ordinary env vars.

# Cross-step state files live on the /workspace volume that Cloud Build shares
# between steps. The defaults are overridable so the scripts can be exercised
# outside Cloud Build (e.g. unit tests) without writing to /workspace.
FAILED_FILE="${FAILED_FILE:-/workspace/FAILED}"
BUILD_VARS_FILE="${BUILD_VARS_FILE:-/workspace/build_vars.env}"

# Record a step id in the failure ledger. The allowFailure provisioning steps
# append here on error so the final check-failure step can fail the build with
# the list of culprits.
record_failure() {
  echo "$1" >> "${FAILED_FILE}"
}

# Short-circuit the rest of a step when an earlier step already failed. Cloud
# Build keeps running later steps after an allowFailure step fails; this turns
# them into no-ops instead of compounding the failure.
skip_if_failed() {
  if [[ -f "${FAILED_FILE}" ]]; then
    echo "Skipping: previous step failed"
    exit 0
  fi
}

# Create a per-run bucket per _BUCKET_TYPE (regional | zonal-RAPID | hns).
create_typed_bucket() {
  local bucket="$1"
  case "${_BUCKET_TYPE}" in
    regional)
      gcloud storage buckets create "gs://$bucket" --project="${PROJECT_ID}" --location="$REGION" ;;
    zonal)
      gcloud storage buckets create "gs://$bucket" --project="${PROJECT_ID}" --location="$REGION" --placement="${_ZONE}" --default-storage-class=RAPID --enable-hierarchical-namespace --uniform-bucket-level-access ;;
    hns)
      gcloud storage buckets create "gs://$bucket" --project="${PROJECT_ID}" --location="$REGION" --enable-hierarchical-namespace --uniform-bucket-level-access ;;
    *)
      echo "ERROR: unknown _BUCKET_TYPE='${_BUCKET_TYPE}' (expected regional|zonal|hns)" >&2
      return 1 ;;
  esac
}

shared_workload_helm_args() {
  local dataset_path="${RUN_DATASET_PATH:-${_DATASET_PATH}}"
  if [ "${_USE_GCSFUSE:-false}" = "true" ]; then
    if [ "${_REUSE_DATASET_BUCKET:-false}" = "true" ]; then
      SRC_OBJECT_PATH=$(echo "${_DATASET_PATH}" | sed -E 's#^gs://[^/]+/?##')
      SRC_OBJECT_PATH="${SRC_OBJECT_PATH%/}"
      if [ -n "${SRC_OBJECT_PATH}" ]; then
        dataset_path="/gcs/dataset/${SRC_OBJECT_PATH}"
      else
        dataset_path="/gcs/dataset"
      fi
    else
      dataset_path="/gcs/dataset"
    fi
  elif [ "${_USE_LUSTRE:-false}" = "true" ] && [ "${_LUSTRE_DATASET_PVC:-lustre-dataset-pvc}" != "none" ]; then
    dataset_path="/lustre/dataset"
  fi
  local gcsfuse_mount_opts="${_GCSFUSE_MOUNT_OPTIONS:-implicit-dirs}"
  if [ "${_GCSFUSE_ENABLE_STREAM_WRITE:-true}" = "true" ]; then
    if [[ "$gcsfuse_mount_opts" != *"write:enable-streaming-writes"* ]]; then
      gcsfuse_mount_opts="${gcsfuse_mount_opts},write:enable-streaming-writes:true"
    fi
    if [[ "$gcsfuse_mount_opts" != *"write:global-max-blocks"* ]]; then
      gcsfuse_mount_opts="${gcsfuse_mount_opts},write:global-max-blocks:-1"
    fi
  else
    if [[ "$gcsfuse_mount_opts" != *"write:enable-streaming-writes"* ]]; then
      gcsfuse_mount_opts="${gcsfuse_mount_opts},write:enable-streaming-writes:false"
    fi
  fi
  if [[ "$gcsfuse_mount_opts" != *"prometheus-port"* ]]; then
    gcsfuse_mount_opts="${gcsfuse_mount_opts},prometheus-port=8080"
  fi
  # Helm's --set / --set-string parses commas as value separators unless escaped with \,
  local helm_gcsfuse_mount_opts="${gcsfuse_mount_opts//,/\\,}"
  local helm_additional_ckpt_paths="${_ADDITIONAL_CHECKPOINT_PATHS//,/\\,}"
  local raw_ts_shape="${_TENSORSTORE_SHAPE:-1000,1000,100}"
  local helm_ts_shape="${raw_ts_shape//,/\\,}"
  local raw_ts_chunks="${_TENSORSTORE_CHUNKS:-100,100,100}"
  local helm_ts_chunks="${raw_ts_chunks//,/\\,}"
  SHARED_HELM_ARGS=(
    --set gcsfs.datasetPath="${dataset_path}"
    --set workload.modelId="${_MODEL_ID}"
    --set-string workload.image="${_IMAGE}"
    --set workload.hfToken="${_HF_TOKEN}"
    --set workload.nodes="${_NODES}"
    --set workload.ranksPerNode="${_RANKS_PER_NODE}"
    --set workload.requirements="${_REQUIREMENTS}"
    --set workload.trainingStrategy="${_TRAINING_STRATEGY}"
    --set workload.tsSingleArray="${_TS_SINGLE_ARRAY:-0}"
    --set workload.parallelCopyWorkers="${_PARALLEL_COPY_WORKERS:-32}"
    --set workload.skipRamdiskStaging="${_SKIP_RAMDISK_STAGING:-1}"
    --set workload.tsDriver="${_TS_DRIVER:-zarr3}"
    --set workload.numShards="${_NUM_SHARDS:-10}"
    --set workload.asyncCheckpoint="${_ASYNC_CHECKPOINT:-false}"
    --set workload.useTensorstore="${_USE_TENSORSTORE:-false}"
    --set workload.additionalCheckpointPaths="${helm_additional_ckpt_paths:-}"
    --set-string workload.tensorstoreShape="${helm_ts_shape}"
    --set-string workload.tensorstoreChunks="${helm_ts_chunks}"
    --set workload.tensorstoreDtype="${_TENSORSTORE_DTYPE:-float32}"
    --set workload.tensorstoreDriver="${_TENSORSTORE_DRIVER:-zarr}"
    --set workload.tensorstoreIterations="${_TENSORSTORE_ITERATIONS:-1}"
    --set workload.tensorstorePerWorkerShape="${_TENSORSTORE_PER_WORKER_SHAPE:-false}"
    --set "nodeSelector.cloud\.google\.com/gke-nodepool=${_MACHINE_TYPE}"
    --set serviceAccount=default
    --set gcsfuse.enabled="${_USE_GCSFUSE:-false}"
    --set gcsfuse.datasetBucket="${DATASET_BUCKET:-}"
    --set gcsfuse.checkpointBucket="${CHECKPOINT_BUCKET:-}"
    --set-string gcsfuse.mountOptions="${helm_gcsfuse_mount_opts}"
    --set-string gcsfuse.sidecarImage="${_GCSFUSE_SIDECAR_IMAGE:-}"
    --set lustre.enabled="${_USE_LUSTRE:-false}"
    --set lustre.datasetPvc="${_LUSTRE_DATASET_PVC:-lustre-dataset-pvc}"
    --set lustre.checkpointPvc="${_LUSTRE_CHECKPOINT_PVC:-lustre-checkpoint-pvc}"
  )
}

setup_lustre_pvcs() {
  if [ "${_USE_LUSTRE:-false}" != "true" ]; then
    return 0
  fi

  local raw_instance="${_LUSTRE_INSTANCE:-}"
  if [ -z "$raw_instance" ]; then
    echo "ERROR: _USE_LUSTRE is true but _LUSTRE_INSTANCE is unset." >&2
    return 1
  fi

  local dataset_pvc="${_LUSTRE_DATASET_PVC:-lustre-dataset-pvc}"
  local checkpoint_pvc="${_LUSTRE_CHECKPOINT_PVC:-lustre-checkpoint-pvc}"
  local capacity="${_LUSTRE_CAPACITY:-12000Gi}"

  local proj="${PROJECT_ID}"
  local zone="${_ZONE}"
  local instance_name="${raw_instance}"

  if [[ "$raw_instance" == projects/* ]]; then
    proj=$(echo "$raw_instance" | cut -d'/' -f2)
    zone=$(echo "$raw_instance" | cut -d'/' -f4)
    instance_name=$(echo "$raw_instance" | cut -d'/' -f6)
  fi

  local ip="${_LUSTRE_IP:-}"
  local fs="${_LUSTRE_FILESYSTEM:-}"

  if [ -z "$ip" ] || [ -z "$fs" ]; then
    local lustre_json
    lustre_json=$(gcloud alpha lustre instances describe "$instance_name" --location="$zone" --project="$proj" --format="json" 2>/dev/null || echo "")
    if [ -n "$lustre_json" ]; then
      local mount_pt
      mount_pt=$(echo "$lustre_json" | jq -r '.mountPoint // empty')
      fs=$(echo "$lustre_json" | jq -r '.filesystem // empty')
      ip=$(echo "$mount_pt" | cut -d'@' -f1)
    fi
  fi

  if [ -z "$ip" ] || [ -z "$fs" ]; then
    echo "ERROR: Could not determine IP or filesystem for Managed Lustre instance ${instance_name} in ${proj}/${zone}." >&2
    echo "Hint: You can provide _LUSTRE_IP and _LUSTRE_FILESYSTEM environment variables (e.g. _LUSTRE_IP=10.50.2.5 _LUSTRE_FILESYSTEM=cltest01)." >&2
    return 1
  fi

  local volume_handle="${proj}/${zone}/${instance_name}"
  echo "--- Configuring Kubernetes PV and PVC for Managed Lustre instance: ${volume_handle} (IP: ${ip}, FS: ${fs}) ---"

  if [ -n "$dataset_pvc" ] && [ "$dataset_pvc" != "none" ]; then
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${dataset_pvc}-pv
spec:
  accessModes:
  - ReadWriteMany
  capacity:
    storage: ${capacity}
  csi:
    driver: lustre.csi.storage.gke.io
    volumeHandle: ${volume_handle}
    volumeAttributes:
      ip: "${ip}"
      filesystem: "${fs}"
  storageClassName: ""
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${dataset_pvc}
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: ${capacity}
  volumeName: ${dataset_pvc}-pv
EOF
  fi

  if [ -n "$checkpoint_pvc" ] && [ "$checkpoint_pvc" != "none" ]; then
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ${checkpoint_pvc}-pv
spec:
  accessModes:
  - ReadWriteMany
  capacity:
    storage: ${capacity}
  csi:
    driver: lustre.csi.storage.gke.io
    volumeHandle: ${volume_handle}
    volumeAttributes:
      ip: "${ip}"
      filesystem: "${fs}"
  storageClassName: ""
  persistentVolumeReclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${checkpoint_pvc}
spec:
  accessModes:
  - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: ${capacity}
  volumeName: ${checkpoint_pvc}-pv
EOF
  fi
}

# Poll a JobSet until it reports Completed (return 0) or Failed/timeout (record
# the failure in the ledger, dump diagnostics, return 1). Shared by the
# seed-checkpoint and run-workload steps so the 240x30s poll lives in one place.
# Usage: wait_for_jobset <jobset-name> <step-id>
wait_for_jobset() {
  local jobset="$1" step="$2" complete failed
  echo "Waiting for JobSet $jobset to complete..."
  for _ in $(seq 1 240); do
    complete=$(kubectl get jobset "$jobset" -o jsonpath='{.status.conditions[?(@.type=="Completed")].status}' 2>/dev/null || echo "")
    failed=$(kubectl get jobset "$jobset" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    if [ "$complete" = "True" ]; then echo "JobSet $jobset completed."; return 0; fi
    if [ "$failed" = "True" ]; then
      echo "JobSet $jobset failed."
      kubectl describe jobset "$jobset" || true
      kubectl get pods -l jobset.sigs.k8s.io/jobset-name="$jobset" -o wide || true
      echo "--- Workload Pod Logs for $jobset ---"
      kubectl logs -l jobset.sigs.k8s.io/jobset-name="$jobset" -c workload --tail=200 2>/dev/null || true
      record_failure "$step"
      return 1
    fi
    sleep 30
  done
  echo "Timed out waiting for JobSet $jobset to complete."
  kubectl describe jobset "$jobset" || true
  kubectl get pods -l jobset.sigs.k8s.io/jobset-name="$jobset" -o wide || true
  echo "--- Workload Pod Logs for $jobset ---"
  kubectl logs -l jobset.sigs.k8s.io/jobset-name="$jobset" -c workload --tail=200 2>/dev/null || true
  record_failure "$step"
  return 1
}
