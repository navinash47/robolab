# Phase 4 test — human gate checklist

**Blunt status:** Phase 4 **code shipped and locally proven** on COMPLETE mlp run `23421c2da07b` (artifact backfill → RecordVideo MP4 H.264 → `wandb.Video` → `GET /api/runs/.../video` 200). **Human gate NOT passed** until you click **Render video** in the UI and watch the robot. Phase 5 (PyBullet) **not started**.

## What was proven (agent)

| Check | Result |
|---|---|
| Local COMPLETE mlp `23421c2da07b` + `checkpoints/.../policy.zip` | OK |
| W&B artifact backfill `policy-23421c2da07b` when missing | OK |
| Fresh env + Gymnasium `RecordVideo` → `videos/.../playback-episode-0.mp4` | OK (~14s, 320×240, **h264**) |
| `wandb.Video` on resumed run `bvwqbhg7` | OK (1 media file) |
| `POST /api/runs/23421c2da07b/render` → `RENDERING` → `READY` | OK (~8s) |
| `GET /api/runs/23421c2da07b/video` | HTTP 200 MP4 |
| RunPod render path | **Not proven** this session (local first; EU-RO-1 capacity flaky) |

## Prereqs

- [ ] `make dev` running (API :8000, web :5173)
- [ ] `WANDB_API_KEY` valid (`W&B: key valid` in header)
- [ ] At least one **COMPLETE** local run with `wandb_url` and preferably `checkpoints/<id>/policy.zip`
  - Known good: **`23421c2da07b`** (mlp wall_follow) or **`f164be140025`**
- [ ] New dep installed: `moviepy` (+ `imageio-ffmpeg`) via `uv sync` — required by Gymnasium RecordVideo

## Exact clicks — local gate (priority)

1. Open http://localhost:5173
2. On Experiments, find a **COMPLETE** mlp `wall_follow` row (prefer `23421c2da07b`)
3. Click **Render video**
4. Expect within ~1–2 minutes:
   - Button shows **Rendering…**
   - Run detail / status shows `video RENDERING` then `READY`
   - Click **Watch** (or open the run name) → HTML5 **video player** appears
   - Play: robot moves in the MuJoCo corridor (wall-follow behavior)
5. Optional: open the run’s **W&B** link → Media / `playback` video also present

## Exact clicks — RunPod run (optional / may be flaky)

1. Only if you have a **COMPLETE** runpod row with a W&B URL and either a local checkpoint copy or a logged `policy-<id>` artifact
2. Same **Render video** button (render runs **on this machine**, not on the dead pod)
3. If it fails with missing checkpoint/artifact: re-train briefly locally or copy `policy.zip` under `checkpoints/<run_id>/`

## Fail modes

| Symptom | Likely cause |
|---|---|
| 400 `Render requires COMPLETE` | Run still training / failed |
| 400 no `wandb_url` | Incomplete train / auth fail |
| `video FAILED` / no MoviePy | `uv sync` missing moviepy — see PHASE_4_APIS.md |
| `MUJOCO_GL=osmesa` error on Mac | Expected — Darwin uses `cgl` (see PHASE_4_APIS conflict) |
| Blank `<video>` but file exists | Rare codec issue; agent smoke used h264 successfully |
| Artifact / checkpoint missing | No local `checkpoints/<id>/policy.zip` and nothing on W&B |

## Gate pass criteria (you confirm)

- [ ] **Render video** on a COMPLETE local run
- [ ] Player appears within a couple of minutes
- [ ] You watch the robot follow the wall
- [ ] (Optional) Same for a COMPLETE RunPod run

Until those boxes are checked: **not gate-passed.** Do **not** start Phase 5 until this gate is human-confirmed.
