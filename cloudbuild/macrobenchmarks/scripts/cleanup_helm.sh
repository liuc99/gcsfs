#!/usr/bin/env bash
# cleanup-helm: uninstall the workload release (best-effort).
if [[ "${_SKIP_CLEANUP}" == "true" ]]; then
  echo "Skipping cleanup-helm as requested."
  exit 0
fi
source "$(dirname "$0")/lib.sh"
source "${BUILD_VARS_FILE}"
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash || true
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="${_ZONE}" --project="${PROJECT_ID}" || true
helm uninstall "$RUN_ID" || true
# Best-effort: the seed release is normally uninstalled by the seed-checkpoint
# step; clean up here in case that step failed mid-way.
helm uninstall "${RUN_ID}-seed" || true
echo "--- Ensuring workload JobSets and pods are deleted ---"
kubectl delete jobset "$RUN_ID" "${RUN_ID}-seed" --ignore-not-found 2>/dev/null || true
kubectl wait --for=delete pod -l "jobset.sigs.k8s.io/jobset-name=${RUN_ID}" --timeout=120s 2>/dev/null || true

