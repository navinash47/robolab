# Phase 6 test / verify

Sim-to-sim transfer (`transfer_to` → report → Transfer tab). Gate sims: **mujoco + pybullet**.

## Preconditions

- API healthy: `curl -s http://127.0.0.1:8000/health` → `{"status":"ok"}`
- Dashboard: `http://localhost:5173`
- A COMPLETE run with local `checkpoints/{run_id}/policy.zip` (e.g. mlp `23421c2da07b` if present)
- Do **not** start 2M trains or Mac Genesis/Isaac

## Unit

```bash
cd /Users/avinashnandyala/Projects/robolab
uv run --project robolab pytest robolab/tests/test_transfer.py -q
```

Expect pass: transfer_ratio / gap / CI helpers.

## Post-hoc transfer (no retrain)

```bash
# Pick a COMPLETE mujoco run that has policy.zip
RUN_ID=23421c2da07b   # or another COMPLETE id

curl -s -X POST "http://127.0.0.1:8000/api/runs/${RUN_ID}/transfer" \
  -H 'Content-Type: application/json' \
  -d '{"targets":["pybullet"],"n_episodes":3}'

# Poll (status RUNNING → READY)
curl -s "http://127.0.0.1:8000/api/runs/${RUN_ID}/transfer" | python -m json.tool | head -60
```

Expect:

- `status: READY`
- `source.sim` = train sim, `targets[0].sim` = `pybullet`
- `targets[0].transfer_ratio` and `gap` present when means exist
- `robustness.friction` / `mass_scale` / `sensor_noise_std` arrays non-empty
- File `transfer/{RUN_ID}/report.json` on disk

## UI gate (human)

1. Open dashboard → **Transfer** tab.
2. Select the COMPLETE run → see table (source vs target, ratio, gap) and three robustness charts.
3. Optional: New Run → check **Transfer to: pybullet** → short smoke (2k) → after COMPLETE, Transfer tab shows auto report.

## Auto-path (train with transfer_to)

```yaml
# configs snippet
sim: mujoco
transfer_to: [pybullet]
trainer:
  timesteps: 2048   # smoke only
```

Or UI New Run with Transfer-to checked. Trainer runs transfer before `/complete` when `transfer_to` is set.

## Non-goals / refuse

- Mac Genesis/Isaac transfer (skipped with reason)
- Auto 2M / overnight pods
- Phase 7 rliable / paper export

## Status

- Code + docs shipped this session.
- Local unit + post-hoc smoke: **PASS** (2026-09-07) — `pytest robolab/tests/test_transfer.py`; POST transfer on `23421c2da07b` → READY, pybullet ratio/gap present, 8 friction sweep points.
- Human UI gate: not passed until Avinash confirms Transfer tab.
