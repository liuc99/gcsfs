#!/usr/bin/env bash
# create-cluster: create the GKE cluster with a small system pool plus a
# dedicated ${_MACHINE_TYPE} pool (whose name the workload's nodeSelector keys
# off), then install the JobSet controller.
set -e
source "$(dirname "$0")/lib.sh"
trap 'record_failure create-cluster' ERR
skip_if_failed
source "${BUILD_VARS_FILE}"

if [ "${IS_EXISTING_NETWORK:-false}" != "true" ]; then
  echo "--- Creating dedicated VPC network: ${NETWORK_NAME} ---"
  gcloud compute networks create "${NETWORK_NAME}" \
    --project="${PROJECT_ID}" \
    --subnet-mode=custom --quiet

  echo "--- Creating dedicated subnetwork: ${SUBNET_NAME} ---"
  gcloud compute networks subnets create "${SUBNET_NAME}" \
    --project="${PROJECT_ID}" \
    --network="${NETWORK_NAME}" \
    --region="${REGION}" \
    --range="10.0.0.0/20" \
    --enable-private-ip-google-access --quiet
else
  echo "--- Using pre-existing VPC network: ${NETWORK_NAME} and subnet: ${SUBNET_NAME} ---"
fi

if gcloud container clusters describe "$CLUSTER_NAME" --zone="${_ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  echo "--- Reusing existing GKE cluster: ${CLUSTER_NAME} in zone ${_ZONE} ---"
  gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="${_ZONE}" --project="${PROJECT_ID}"
  if ! gcloud container node-pools describe "${_MACHINE_TYPE}" --cluster="$CLUSTER_NAME" --zone="${_ZONE}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "--- Creating dedicated node pool ${_MACHINE_TYPE} on existing cluster ${CLUSTER_NAME} ---"
    NODE_POOL_ARGS=(
      --cluster="$CLUSTER_NAME"
      --project="${PROJECT_ID}"
      --zone="${_ZONE}"
      --machine-type="${_MACHINE_TYPE}"
      --num-nodes="${_NODES}"
      --disk-size="200"
      --disk-type="hyperdisk-balanced"
      --enable-gvnic
      --service-account="${_GKE_SERVICE_ACCOUNT}"
      --scopes="https://www.googleapis.com/auth/cloud-platform"
      --no-enable-autoupgrade
      --quiet
    )
    if [ "${_ENABLE_TIER1_NETWORKING:-true}" = "true" ]; then
      NODE_POOL_ARGS+=(--network-performance-configs="total-egress-bandwidth-tier=TIER_1")
    fi
    if [ -n "${_RESERVATION_AFFINITY}" ]; then
      NODE_POOL_ARGS+=(--reservation-affinity="${_RESERVATION_AFFINITY}")
    fi
    if [ -n "${_RESERVATION_NAME}" ]; then
      NODE_POOL_ARGS+=(--reservation="${_RESERVATION_NAME}")
    fi
    gcloud container node-pools create "${_MACHINE_TYPE}" "${NODE_POOL_ARGS[@]}"
  else
    echo "--- Node pool ${_MACHINE_TYPE} already exists on cluster ${CLUSTER_NAME}, ensuring size is ${_NODES} ---"
    gcloud container clusters resize "$CLUSTER_NAME" --node-pool="${_MACHINE_TYPE}" --num-nodes="${_NODES}" --zone="${_ZONE}" --project="${PROJECT_ID}" --quiet || true
  fi
else
  echo "--- Creating GKE cluster: ${CLUSTER_NAME} ---"
  gcloud container clusters create "$CLUSTER_NAME" \
    --project="${PROJECT_ID}" --zone="${_ZONE}" \
    --machine-type="e2-standard-4" --num-nodes="1" \
    --service-account="${_GKE_SERVICE_ACCOUNT}" \
    --scopes="https://www.googleapis.com/auth/cloud-platform" \
    --private-ipv6-google-access-type=outbound-only \
    --network="${NETWORK_NAME}" --subnetwork="${SUBNET_NAME}" \
    --addons=GcsFuseCsiDriver,LustreCsiDriver \
    --workload-pool="${PROJECT_ID}.svc.id.goog" \
    --no-enable-autoupgrade --quiet
  NODE_POOL_ARGS=(
    --cluster="$CLUSTER_NAME"
    --project="${PROJECT_ID}"
    --zone="${_ZONE}"
    --machine-type="${_MACHINE_TYPE}"
    --num-nodes="${_NODES}"
    --disk-size="200"
    --disk-type="hyperdisk-balanced"
    --enable-gvnic
    --service-account="${_GKE_SERVICE_ACCOUNT}"
    --scopes="https://www.googleapis.com/auth/cloud-platform"
    --no-enable-autoupgrade
    --quiet
  )
  if [ "${_ENABLE_TIER1_NETWORKING:-true}" = "true" ]; then
    NODE_POOL_ARGS+=(--network-performance-configs="total-egress-bandwidth-tier=TIER_1")
  fi
  if [ -n "${_RESERVATION_AFFINITY}" ]; then
    NODE_POOL_ARGS+=(--reservation-affinity="${_RESERVATION_AFFINITY}")
  fi
  if [ -n "${_RESERVATION_NAME}" ]; then
    NODE_POOL_ARGS+=(--reservation="${_RESERVATION_NAME}")
  fi
  gcloud container node-pools create "${_MACHINE_TYPE}" "${NODE_POOL_ARGS[@]}"
  gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="${_ZONE}" --project="${PROJECT_ID}"
fi
echo "--- Configuring Workload Identity for default service account ---"
gcloud iam service-accounts add-iam-policy-binding "${_GKE_SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/default]" \
  --quiet || true
kubectl annotate serviceaccount default "iam.gke.io/gcp-service-account=${_GKE_SERVICE_ACCOUNT}" --overwrite
gcloud artifacts repositories add-iam-policy-binding benchmarks \
  --location="${REGION:-us-central1}" \
  --project="${PROJECT_ID}" \
  --member="serviceAccount:${_GKE_SERVICE_ACCOUNT}" \
  --role="roles/artifactregistry.reader" \
  --quiet || true
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${_JOBSET_VERSION}/manifests.yaml"
kubectl rollout status deployment/jobset-controller-manager -n jobset-system --timeout=300s
wait_for_resource_creation() {
  local kind_name="$1"
  local namespace="$2"
  local timeout=60
  local elapsed=0
  until kubectl get "$kind_name" -n "$namespace" >/dev/null 2>&1 || [ "$elapsed" -ge "$timeout" ]; do
    sleep 2
    elapsed=$((elapsed + 2))
  done
}

if [ "${_USE_GCSFUSE:-false}" = "true" ]; then
  echo "--- Waiting for GCSFuse CSI driver node daemonset to be ready ---"
  wait_for_resource_creation daemonset/gcsfusecsi-node kube-system
  kubectl rollout status daemonset/gcsfusecsi-node -n kube-system --timeout=300s || true
fi
if [ "${_USE_LUSTRE:-false}" = "true" ]; then
  echo "--- Waiting for Managed Lustre CSI driver node daemonset to be ready ---"
  ds_name=""
  for _ in $(seq 1 60); do
    ds_name=$(kubectl get daemonset -n kube-system -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | tr ' ' '\n' | grep -i lustre | head -1 || echo "")
    if [ -n "$ds_name" ]; then
      break
    fi
    sleep 2
  done
  if [ -n "$ds_name" ]; then
    echo "Found Lustre CSI daemonset: ${ds_name}"
    kubectl rollout status "daemonset/${ds_name}" -n kube-system --timeout=300s || true
  else
    echo "WARNING: Could not find Lustre CSI daemonset in kube-system; proceeding with setup_lustre_pvcs."
  fi
  setup_lustre_pvcs
fi
