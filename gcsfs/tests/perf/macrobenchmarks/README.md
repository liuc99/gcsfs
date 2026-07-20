# Macrobenchmarks Execution Guide

This guide provides step-by-step instructions for running macrobenchmarks (such as `tensorstore-gcsfuse` and `hf-pytorch-lightning-cpu`) directly using standard open-source tools (`gcloud`, `kubectl`, and `helm`).

---

## 1. Prerequisites & Required Tools

Ensure you have the following installed and configured on your local system:
* **[Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install)**: Authenticated to your GCP project (`gcloud auth login`).
* **[kubectl](https://kubernetes.io/docs/tasks/tools/)**: Kubernetes command-line tool.
* **[Helm 3](https://helm.sh/docs/intro/install/)**: Kubernetes package manager.

---

## 2. Environment Setup

Configure your shell environment variables:

```bash
# 1. GCP Project & Region Configuration
export PROJECT_ID="$(gcloud config get-value project)"
export REGION="us-central1"
export ZONE="us-central1-a"

# 2. Resource Naming
export INFRA_PREFIX="perf-test"
export RUN_ID="${INFRA_PREFIX}-run-$(date +%s)"

# 3. GKE Service Account
export GKE_SERVICE_ACCOUNT="my-gke-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# 4. GCS Bucket Names
export CHECKPOINT_BUCKET="${INFRA_PREFIX}-ckpt-${RUN_ID}"
```

> **Note:** If your GCP Service Account does not exist yet, create it and grant Storage Admin permissions:
> ```bash
> gcloud iam service-accounts create my-gke-sa --project="${PROJECT_ID}"
> gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
>   --member="serviceAccount:${GKE_SERVICE_ACCOUNT}" \
>   --role="roles/storage.admin"
> ```

---

## 3. Storage Setup (GCS Bucket)

Create a region-co-located GCS bucket for benchmark checkpoints:

```bash
gcloud storage buckets create "gs://${CHECKPOINT_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
```

---

## 4. GKE Cluster & Infrastructure Provisioning

### 4.1 Create GKE Cluster with GCSFuse CSI Driver
Create a standard GKE cluster with the GCSFuse CSI plugin and Workload Identity enabled:

```bash
export CLUSTER_NAME="${INFRA_PREFIX}-cluster"

gcloud container clusters create "${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="e2-standard-4" \
  --num-nodes=1 \
  --service-account="${GKE_SERVICE_ACCOUNT}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --addons=GcsFuseCsiDriver \
  --workload-pool="${PROJECT_ID}.svc.id.goog" \
  --no-enable-autoupgrade --quiet

# Obtain cluster credentials for kubectl
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}"
```

### 4.2 Create Node Pool
Create a dedicated node pool matching your workload machine requirements (e.g., `n2-standard-8` or high-performance types like `c4-standard-192`):

```bash
export MACHINE_TYPE="n2-standard-8"
export NODES=1

gcloud container node-pools create "${MACHINE_TYPE}" \
  --cluster="${CLUSTER_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --num-nodes="${NODES}" \
  --disk-size=200 \
  --disk-type="pd-balanced" \
  --enable-gvnic \
  --service-account="${GKE_SERVICE_ACCOUNT}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --no-enable-autoupgrade --quiet
```

---

### 4.3 Configure Workload Identity & Install JobSet Controller

Configure Workload Identity so Kubernetes pods using the `default` ServiceAccount authenticate as your GCP Service Account:

```bash
gcloud iam service-accounts add-iam-policy-binding "${GKE_SERVICE_ACCOUNT}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${PROJECT_ID}.svc.id.goog[default/default]"

kubectl annotate serviceaccount default "iam.gke.io/gcp-service-account=${GKE_SERVICE_ACCOUNT}" --overwrite
```

Install the open-source Kubernetes `JobSet` controller:

```bash
JOBSET_VERSION="v0.12.0"
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/jobset/releases/download/${JOBSET_VERSION}/manifests.yaml"
kubectl rollout status deployment/jobset-controller-manager -n jobset-system --timeout=300s
```

---

## 5. Deploy & Run Benchmark via Helm

From the repository root (or `gcsfs/tests/perf/macrobenchmarks`), install the workload Helm chart:

```bash
WORKLOAD="tensorstore-gcsfuse"
CHART_DIR="workloads/${WORKLOAD}/helm_chart"

helm install "${RUN_ID}" "${CHART_DIR}" \
  -f "${CHART_DIR}/values_base.yaml" \
  --set gcsfuse.enabled=true \
  --set gcsfuse.checkpointBucket="${CHECKPOINT_BUCKET}" \
  --set gcsfs.ckptWritePath="/gcs/checkpoints/checkpoints" \
  --set workload.steps=1 \
  --set workload.tensorstoreShape="1000,1000,100" \
  --set workload.tensorstoreChunks="100,100,100" \
  --set nodeSelector."cloud\.google\.com/gke-nodepool"="${MACHINE_TYPE}"
```

---

## 6. Monitor Workload & Retrieve Results

### 6.1 Monitor Job Completion
Check the execution status of the JobSet:

```bash
kubectl get jobset "${RUN_ID}" -w
```

### 6.2 View Workload Logs
Inspect pod logs to view benchmark metrics and performance results:

```bash
kubectl logs -l jobset.sigs.k8s.io/jobset-name="${RUN_ID}" -c workload --tail=1000
```

---

## 7. Resource Cleanup

Delete all provisioned cloud resources after testing to avoid ongoing charges:

```bash
# 1. Uninstall Helm release
helm uninstall "${RUN_ID}"

# 2. Delete GKE Cluster
gcloud container clusters delete "${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --quiet

# 3. Delete GCS Bucket
gcloud storage rm -r "gs://${CHECKPOINT_BUCKET}"
```
