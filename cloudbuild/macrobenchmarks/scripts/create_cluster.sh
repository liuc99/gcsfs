#!/usr/bin/env bash
# create-cluster: create the GKE cluster with a small system pool plus a
# dedicated ${_MACHINE_TYPE} pool (whose name the workload's nodeSelector keys
# off), then install the JobSet controller.
set -e
source "$(dirname "$0")/lib.sh"
trap 'record_failure create-cluster' ERR
skip_if_failed
source "${BUILD_VARS_FILE}"

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

# `gcloud container clusters create` has no --node-pool flag and always names
# its initial pool "default-pool", which would not match the workload's
# nodeSelector (cloud.google.com/gke-nodepool=c4-standard-192). Create a small
# system pool for the cluster, then a dedicated ${_MACHINE_TYPE} pool whose name
# the GKE node label (and the chart's nodeSelector) keys off of. --enable-gvnic
# puts the nodes on gVNIC, which C4 requires and which TIER_1 egress (the direct
# high-bandwidth path) is gated on.
gcloud container clusters create "$CLUSTER_NAME" \
  --project="${PROJECT_ID}" --zone="${_ZONE}" \
  --machine-type="e2-standard-4" --num-nodes="1" \
  --service-account="${_GKE_SERVICE_ACCOUNT}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --private-ipv6-google-access-type=outbound-only \
  --network="${NETWORK_NAME}" --subnetwork="${SUBNET_NAME}" \
  --addons=GcsFuseCsiDriver \
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
echo "--- Configuring Workload Identity for default service account ---"
gcloud iam service-accounts add-iam-policy-binding "${_GKE_SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/default]" \
  --quiet || true
kubectl annotate serviceaccount default "iam.gke.io/gcp-service-account=${_GKE_SERVICE_ACCOUNT}" --overwrite
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${_JOBSET_VERSION}/manifests.yaml"
kubectl rollout status deployment/jobset-controller-manager -n jobset-system --timeout=300s
if [ "${_USE_GCSFUSE:-false}" = "true" ]; then
  echo "--- Waiting for GCSFuse CSI driver sidecar injector and node driver to be ready ---"
  kubectl rollout status deployment/gke-gcsfuse-sidecar-injector -n kube-system --timeout=300s || true
  kubectl rollout status daemonset/gke-gcsfuse-node -n kube-system --timeout=300s || true
fi
