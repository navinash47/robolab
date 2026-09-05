#!/usr/bin/env bash
# Thin image bootstrap: clone GIT_SHA, then exec the repo entrypoint so
# entrypoint.sh fixes ship without rebuilding the worker image.
set -euo pipefail

log() { echo "[robolab-bootstrap] $*"; }

: "${RUN_ID:?RUN_ID required}"
: "${GIT_SHA:?GIT_SHA required}"
: "${BACKEND_URL:?BACKEND_URL required}"
: "${ROBOLAB_GIT_URL:?ROBOLAB_GIT_URL required}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required}"

report_fail() {
  local msg="$1"
  local escaped
  escaped=$(printf '%s' "${msg}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
    || printf '"%s"' "${msg}")
  curl -sS --max-time 30 -X POST "${BACKEND_URL}/api/runs/${RUN_ID}/fail" \
    -H "Content-Type: application/json" \
    -d "{\"error\": ${escaped}}" || true
}

terminate() {
  if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    curl -sS -X DELETE "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
    curl -sS -X POST "https://api.runpod.io/graphql" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" || true
  fi
}

on_err() {
  local code=$?
  report_fail "bootstrap failed (exit=${code}) before repo entrypoint"
  terminate
  exit "${code}"
}
trap on_err ERR

REPO_DIR="${REPO_DIR:-/workspace/robolab}"
mkdir -p /workspace/.cache/uv

CLONE_URL="${ROBOLAB_GIT_URL}"
if [[ -n "${GITHUB_TOKEN:-}" && "${CLONE_URL}" =~ ^https://github.com/ ]]; then
  CLONE_URL="https://x-access-token:${GITHUB_TOKEN}@${CLONE_URL#https://}"
fi

if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Updating existing clone"
  git -C "${REPO_DIR}" remote set-url origin "${CLONE_URL}" || true
  git -C "${REPO_DIR}" fetch --all --tags || true
  git -C "${REPO_DIR}" checkout --force "${GIT_SHA}"
else
  log "Cloning ${ROBOLAB_GIT_URL} @ ${GIT_SHA}"
  rm -rf "${REPO_DIR}"
  git clone "${CLONE_URL}" "${REPO_DIR}"
  git -C "${REPO_DIR}" checkout --force "${GIT_SHA}"
fi

ENTRY="${REPO_DIR}/robolab/robolab/compute/entrypoint.sh"
if [[ ! -x "${ENTRY}" ]]; then
  chmod +x "${ENTRY}" || true
fi
if [[ ! -f "${ENTRY}" ]]; then
  report_fail "repo entrypoint missing at ${ENTRY}"
  terminate
  exit 1
fi

# Disable ERR trap — entrypoint has its own EXIT cleanup.
trap - ERR
log "Exec repo entrypoint ${ENTRY}"
exec bash "${ENTRY}"
