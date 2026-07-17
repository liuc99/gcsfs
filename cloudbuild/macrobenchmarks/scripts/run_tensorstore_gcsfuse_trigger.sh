#!/usr/bin/env bash
# Helper script to submit Cloud Build trigger for TensorStore + GCSFuse benchmark
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo '')}"
LOCATION="${LOCATION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
INFRA_PREFIX="${INFRA_PREFIX:-tensorstore-fuse-perf}"
GKE_SERVICE_ACCOUNT="${GKE_SERVICE_ACCOUNT:-}"

if [ -z "$PROJECT_ID" ]; then
  echo "Error: PROJECT_ID is not set and could not be detected from gcloud config." >&2
  exit 1
fi

if [ -z "$GKE_SERVICE_ACCOUNT" ]; then
  echo "Usage: GKE_SERVICE_ACCOUNT=<sa_email> INFRA_PREFIX=<prefix> ZONE=<zone> $0"
  echo "Example:"
  echo "  GKE_SERVICE_ACCOUNT=gke-sa@my-project.iam.gserviceaccount.com INFRA_PREFIX=ts-gcsfuse-test ZONE=us-central1-a ./run_tensorstore_gcsfuse_trigger.sh"
  exit 1
fi

echo "=================================================="
echo " Submitting TensorStore + GCSFuse Cloud Build Job "
echo "=================================================="
echo " Project ID          : $PROJECT_ID"
echo " Location            : $LOCATION"
echo " Zone                : $ZONE"
echo " Infra Prefix        : $INFRA_PREFIX"
echo " GKE Service Account : $GKE_SERVICE_ACCOUNT"
echo " Config File         : cloudbuild/macrobenchmarks/macrobenchmarks-tensorstore-gcsfuse-cloudbuild.yaml"
echo "=================================================="

gcloud builds submit \
  --config=cloudbuild/macrobenchmarks/macrobenchmarks-tensorstore-gcsfuse-cloudbuild.yaml \
  --substitutions="_INFRA_PREFIX=${INFRA_PREFIX},_ZONE=${ZONE},_GKE_SERVICE_ACCOUNT=${GKE_SERVICE_ACCOUNT}" \
  --project="$PROJECT_ID"
