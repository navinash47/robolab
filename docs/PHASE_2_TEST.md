# Phase 2 test — human gate checklist

**Blunt status:** Code path proven end-to-end (kan COMPLETE smoke + Compare pulls **real** W&B curves for mlp `23421c2da07b` / `bvwqbhg7` and kan `ffd68ba7639a` / `5yu1fkqd`). **Human gate not passed until you click through the UI below.** Phase 3 (RunPod) was **not** started.

## Prereqs

1. `make dev` running → http://localhost:5173 + API :8000
2. Header shows **W&B: key valid** (`auth_ok`)
3. `.env` has `WANDB_API_KEY` + `WANDB_PROJECT=robolab` (already used in Phase 1)
4. **No RunPod key needed** for Phase 2

## Recommended timesteps (Mac CPU)

| Arch | Form option | Notes |
|---|---|---|
| `mlp` | 50k | Already proven COMPLETE (`23421c2da07b`) |
| `kan` | **5k (smoke)** for gate | 2k smoke completed in ~10s locally; **50k still in the form** but expect many minutes on CPU |
| `kan` | 2k (quick) | Fine for wiring proof |

## Exact clicks — kan run

1. Open http://localhost:5173
2. Click **New Run**
3. Compute: **local**
4. Arch: **kan** (dropdown must list `mlp` and `kan`)
5. Task: **wall_follow**
6. Timesteps: **5k (smoke)** (or 2k if impatient; 50k if you want a long overnight curve)
7. Click **Start**
8. **Expect:** row appears `RUNNING`, progress bar advances, mean return updates, **W&B** link appears, status → **COMPLETE**
9. Open the W&B link → see `rollout/ep_rew_mean` curve (real, not mock)

**Already proven smoke (API):** run `ffd68ba7639a`, arch `kan`, param_count `23680`, wandb `https://wandb.ai/avinashnandyala2-umass-amherst/robolab/runs/5yu1fkqd`, COMPLETE.

## Exact clicks — Compare (mlp + kan)

1. On **Experiments**, check the boxes for:
   - COMPLETE mlp: `23421c2da07b` (or any COMPLETE mlp with W&B)
   - COMPLETE kan: your new kan run (or `ffd68ba7639a`)
2. Click **Compare selected (N)** — or open the **Compare** nav tab after loading
3. **Expect UI:**
   - One Recharts chart with **two lines** (mlp + kan) vs step
   - Table with columns: Run, Arch, **Param count**, Final mean return, History pts, W&B
   - mlp param count ~9088 (computed); kan ~23680 (logged)
4. Curves must come from W&B API (`POST /api/compare`) — empty chart + per-row error means key/network/history failure, **not** a mock fallback

## Fail modes

| Symptom | Likely cause |
|---|---|
| Arch dropdown missing `kan` | API not reloaded / `efficient-kan` missing — `uv sync --all-packages`, restart `make dev` |
| kan run FAIL at import | `efficient-kan` install broke — check `uv run --package robolab python -c "from efficient_kan import KAN"` |
| W&B: key rejected | Bad/truncated `WANDB_API_KEY` — fresh key from https://wandb.ai/authorize |
| Compare chart empty / history pts = 0 | Run has no `wandb_url`, or metric `rollout/ep_rew_mean` never logged, or W&B API timeout |
| Compare 400 WANDB_API_KEY missing | API process missing env — restart `make dev` with `.env` |
| kan 50k “stuck” | Slow CPU — wait; use 5k for gate; progress should still heartbeat |
| Param count "—" | Old mlp run before Phase 2 — Compare recomputes from `arch_cfg`; if still blank, arch registry import failed |

## API smoke (optional)

```bash
curl -s http://127.0.0.1:8000/api/archs
# → {"archs":["kan","mlp"]}

curl -s -X POST http://127.0.0.1:8000/api/compare \
  -H 'Content-Type: application/json' \
  -d '{"run_ids":["23421c2da07b","ffd68ba7639a"]}'
# → two runs, non-empty history arrays, param_count set
```

## Gate pass criteria (you confirm)

- [ ] Launched a **kan** run from the UI the same way as mlp
- [ ] Opened **Compare**, selected mlp + kan
- [ ] Saw **two curves** on one chart
- [ ] Saw table with **param counts** for both

Until those boxes are checked in the browser: **not gate-passed.**
