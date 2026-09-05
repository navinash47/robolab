#!/usr/bin/env bash
# RoboLab worker entrypoint — always self-terminates the RunPod on EXIT.
set -euo pipefail

log() { echo "[robolab-entrypoint] $*"; }

cleanup() {
  local code=$?
  log "EXIT trap (code=${code}) — terminating pod ${RUNPOD_POD_ID:-<unset>}"
  if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
    # Prefer REST v2 (api.runpod.io/v2); fall back to documented v1 Manage Pods path.
    curl -sS -X DELETE \
      "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      || curl -sS -X DELETE \
        "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      || log "WARN: terminate request failed (pod may already be gone)"
  else
    log "WARN: RUNPOD_POD_ID or RUNPOD_API_KEY missing — cannot self-terminate"
  fi
  exit "${code}"
}
trap cleanup EXIT

: "${RUN_ID:?RUN_ID required}"
: "${GIT_SHA:?GIT_SHA required}"
: "${CONFIG_B64:?CONFIG_B64 required}"
: "${BACKEND_URL:?BACKEND_URL required}"
: "${ROBOLAB_GIT_URL:?ROBOLAB_GIT_URL required}"
: "${WANDB_API_KEY:?WANDB_API_KEY required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required}"

export WANDB_PROJECT="${WANDB_PROJECT:-robolab}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

REPO_DIR="${REPO_DIR:-/workspace/robolab}"
RUN_DIR="/workspace/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" /workspace/.cache

# Optional private clone
CLONE_URL="${ROBOLAB_GIT_URL}"
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  # https://github.com/org/repo.git → https://x-access-token:TOKEN@github.com/org/repo.git
  if [[ "${CLONE_URL}" =~ ^https://github.com/ ]]; then
    CLONE_URL="https://x-access-token:${GITHUB_TOKEN}@${CLONE_URL#https://}"
  fi
fi

if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Fetching existing clone at ${REPO_DIR}"
  git -C "${REPO_DIR}" remote set-url origin "${CLONE_URL}" || true
  git -C "${REPO_DIR}" fetch --all --tags || true
  git -C "${REPO_DIR}" checkout --force "${GIT_SHA}"
else
  log "Cloning ${ROBOLAB_GIT_URL} @ ${GIT_SHA}"
  rm -rf "${REPO_DIR}"
  git clone "${CLONE_URL}" "${REPO_DIR}"
  git -C "${REPO_DIR}" checkout --force "${GIT_SHA}"
fi

cd "${REPO_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

log "uv sync"
uv sync --all-packages --python 3.11

CONFIG_PATH="${RUN_DIR}/config.yaml"
python3 - <<PY
import base64, pathlib
raw = base64.b64decode("${CONFIG_B64}")
path = pathlib.Path("${CONFIG_PATH}")
path.write_bytes(raw)
print(f"wrote {path} ({len(raw)} bytes)")
PY

# Prefer CUDA on the pod
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log "Starting trainer run_id=${RUN_ID}"
uv run --package robolab python -m robolab.train.trainer \
  --run-id "${RUN_ID}" \
  --config "${CONFIG_PATH}" \
  --backend-url "${BACKEND_URL}"

log "Trainer finished OK"
