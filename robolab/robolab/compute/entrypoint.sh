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

# Optional heavy sims: install on the network volume cache (no Mac; no default image bloat).
# Parse sim without PyYAML (may be unavailable before uv sync).
SIM_NAME=$(
  CONFIG_PATH="${CONFIG_PATH}" python3 - <<'PY'
import json, os, pathlib, re
raw = pathlib.Path(os.environ["CONFIG_PATH"]).read_text()
# Config is JSON from model_dump_json() base64; still accept YAML-ish `sim:`.
try:
    print((json.loads(raw).get("sim") or "").strip())
except Exception:
    m = re.search(r'(?m)^\s*sim:\s*["\']?([A-Za-z0-9_]+)', raw)
    print(m.group(1) if m else "")
PY
)
if [[ "${SIM_NAME}" == "genesis" || "${ROBOLAB_INSTALL_GENESIS:-}" == "1" ]]; then
  log "Installing genesis-world (optional; cached under ${UV_CACHE_DIR})"
  export ROBOLAB_GENESIS_GPU="${ROBOLAB_GENESIS_GPU:-1}"
  if ! uv pip install --python 3.11 genesis-world 2>/tmp/robolab-genesis-install.err; then
    # Prefer project venv after sync
    if ! uv run --package robolab pip install genesis-world 2>>/tmp/robolab-genesis-install.err; then
      tail -c 1200 /tmp/robolab-genesis-install.err > /tmp/robolab-genesis-install.tail || true
      report_fail "genesis-world install failed: $(tr '\n' ' ' </tmp/robolab-genesis-install.tail | tr -cd '[:print:] ')"
      exit 1
    fi
  fi
fi
if [[ "${SIM_NAME}" == "isaaclab" || "${SIM_NAME}" == "isaac_sim" ]]; then
  # Isaac is multi-GB — not auto-installed. Fail fast with pointer (adapter is still a stub).
  report_fail "sim=${SIM_NAME} requires NVIDIA Isaac on Linux GPU. This worker image does not bake Isaac (multi-GB). See docs/ISAAC_INSTALL.md — approve a separate :isaac image before rebuild. Mac cannot run Isaac."
  exit 1
fi

log "Ensuring xvfb/glfw for headless MuJoCo"
if ! command -v xvfb-run >/dev/null 2>&1 || ! ldconfig -p 2>/dev/null | grep -qi glfw; then
  apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      xvfb libglfw3 libglfw3-dev libosmesa6 libgl1-mesa-glx libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/* \
    || log "WARN: xvfb/glfw apt install failed"
fi
# GLFW+xvfb is the most reliable headless path on Secure GPU hosts.
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
log "MUJOCO_GL=${MUJOCO_GL}"

# Prefer CUDA on the pod, but probe first — a bad driver can SIGSEGV torch.
log "Probing CUDA in a subprocess (segfaults must not kill the trainer)"
set +e
uv run --package robolab python -c 'import torch; assert torch.cuda.is_available(); torch.zeros(1, device="cuda"); print("cuda_ok")' \
  >/tmp/robolab-cuda-probe.out 2>/tmp/robolab-cuda-probe.err
cuda_rc=$?
set -e
if [[ "${cuda_rc}" -ne 0 ]] || ! grep -q cuda_ok /tmp/robolab-cuda-probe.out 2>/dev/null; then
  log "CUDA probe failed (rc=${cuda_rc}) — forcing CUDA_VISIBLE_DEVICES= and device=cpu"
  export CUDA_VISIBLE_DEVICES=""
  if [[ -f "${CONFIG_PATH}" ]]; then
    python3 -c "import pathlib,re,sys; p=pathlib.Path(sys.argv[1]); t=p.read_text(); t=re.sub(r'\"device\"\\s*:\\s*\"cuda\"', '\"device\": \"cpu\"', t); t=re.sub(r'device:\\s*cuda', 'device: cpu', t); p.write_text(t); print('rewrote', p)" \
      "${CONFIG_PATH}"
  fi
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  log "CUDA probe OK"
fi

log "Starting trainer run_id=${RUN_ID}"
set +e
TRAIN_CMD=(uv run --package robolab python -m robolab.train.trainer \
  --run-id "${RUN_ID}" \
  --config "${CONFIG_PATH}" \
  --backend-url "${BACKEND_URL}")
if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a -s "-screen 0 640x480x24" "${TRAIN_CMD[@]}" \
    > /tmp/robolab-trainer.out 2> /tmp/robolab-trainer.err
else
  "${TRAIN_CMD[@]}" > /tmp/robolab-trainer.out 2> /tmp/robolab-trainer.err
fi
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
