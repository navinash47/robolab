# RoboLab worker entrypoint — always self-terminates the RunPod on EXIT.
set -euo pipefail

log() { echo "[robolab-entrypoint] $*"; }

TRAINER_OK=0
FAIL_POSTED=0

report_fail() {
  local msg="$1"
  [[ "${FAIL_POSTED}" -eq 1 ]] && return 0
  FAIL_POSTED=1
  if [[ -z "${BACKEND_URL:-}" || -z "${RUN_ID:-}" ]]; then
    log "WARN: cannot POST fail (BACKEND_URL/RUN_ID unset): ${msg}"
    return 0
  fi
  # JSON-escape message without relying on a full venv
  local escaped
  escaped=$(printf '%s' "${msg}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
    || printf '"%s"' "${msg//\"/\\\"}")
  curl -sS --max-time 30 -X POST "${BACKEND_URL}/api/runs/${RUN_ID}/fail" \
    -H "Content-Type: application/json" \
    -d "{\"error\": ${escaped}}" \
    && log "Posted fail to backend" \
    || log "WARN: fail POST did not succeed"
}

cleanup() {
  local code=$?
  # Do not let set -e abort mid-cleanup; always attempt terminate.
  set +e
  if [[ "${TRAINER_OK}" -ne 1 && "${code}" -ne 0 ]]; then
    report_fail "entrypoint exited code=${code} before trainer completed (see pod logs: git clone / uv sync / trainer)"
  fi
  log "EXIT trap (code=${code}) — terminating pod ${RUNPOD_POD_ID:-<unset>}"
  if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
    local ok=0
    local http
    # Prefer REST v2; then legacy v1; then GraphQL (some keys 403 on REST DELETE).
    http=$(curl -sS -o /tmp/robolab-term.out -w "%{http_code}" -X DELETE \
      "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
      -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true)
    if [[ "${http}" =~ ^(200|204|404|410)$ ]]; then
      log "Terminated via REST v2 (HTTP ${http})"; ok=1
    else
      http=$(curl -sS -o /tmp/robolab-term.out -w "%{http_code}" -X DELETE \
        "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true)
      if [[ "${http}" =~ ^(200|204|404|410)$ ]]; then
        log "Terminated via REST v1 (HTTP ${http})"; ok=1
      else
        local gql
        gql=$(curl -sS -X POST "https://api.runpod.io/graphql" \
          -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
          -H "Content-Type: application/json" \
          --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" || true)
        if echo "${gql}" | grep -Eq 'podTerminate|null|"id"'; then
          log "Terminated via GraphQL"; ok=1
        fi
      fi
    fi
    if [[ "${ok}" -ne 1 ]]; then
      log "WARN: terminate failed (REST ${http:-?}); watchdog should reclaim"
    fi
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
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"

REPO_DIR="${REPO_DIR:-/workspace/robolab}"
RUN_DIR="/workspace/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" /workspace/.cache/uv

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
  git -C "${REPO_DIR}" checkout --force "${GIT_SHA}" \
    || { report_fail "git checkout ${GIT_SHA} failed in existing clone"; exit 1; }
else
  log "Cloning ${ROBOLAB_GIT_URL} @ ${GIT_SHA}"
  rm -rf "${REPO_DIR}"
  git clone "${CLONE_URL}" "${REPO_DIR}" \
    || { report_fail "git clone failed for ${ROBOLAB_GIT_URL}"; exit 1; }
  git -C "${REPO_DIR}" checkout --force "${GIT_SHA}" \
    || { report_fail "git checkout ${GIT_SHA} failed after clone"; exit 1; }
fi

cd "${REPO_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh \
    || { report_fail "uv install script failed"; exit 1; }
  export PATH="${HOME}/.local/bin:${PATH}"
fi

log "uv sync (cache=${UV_CACHE_DIR})"
if ! uv sync --all-packages --python 3.11 2>/tmp/robolab-uv-sync.err; then
  tail -c 1500 /tmp/robolab-uv-sync.err > /tmp/robolab-uv-sync.tail || true
  report_fail "uv sync failed: $(tr '\n' ' ' </tmp/robolab-uv-sync.tail | tr -cd '[:print:] ')"
  exit 1
fi

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
set +e
uv run --package robolab python -m robolab.train.trainer \
  --run-id "${RUN_ID}" \
  --config "${CONFIG_PATH}" \
  --backend-url "${BACKEND_URL}" \
  > /tmp/robolab-trainer.out 2> /tmp/robolab-trainer.err
trainer_rc=$?
set -e
if [[ -s /tmp/robolab-trainer.out ]]; then
  log "trainer stdout (tail):"; tail -n 40 /tmp/robolab-trainer.out || true
fi
if [[ -s /tmp/robolab-trainer.err ]]; then
  log "trainer stderr (tail):"; tail -n 40 /tmp/robolab-trainer.err || true
fi
if [[ "${trainer_rc}" -ne 0 ]]; then
  detail=$(tr '\n' ' ' </tmp/robolab-trainer.err 2>/dev/null | tr -cd '[:print:] ' | tail -c 1500)
  FAIL_POSTED=0
  report_fail "trainer rc=${trainer_rc}: ${detail:-no stderr captured}"
  exit "${trainer_rc}"
fi

TRAINER_OK=1
log "Trainer finished OK"
