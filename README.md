# RoboLab

Standalone robotics RL lab: train, evaluate, transfer, and visualize runs from a dashboard.

## Quick start

```bash
cp .env.example .env
# Set WANDB_API_KEY from https://wandb.ai/authorize
# WANDB_PROJECT=robolab
make dev
```

- Web: http://localhost:5173
- API: http://127.0.0.1:8000

## Phase 1

Local MuJoCo + MLP + SB3 PPO + W&B + live progress (SSE).

Human gate: **New Run → local → mlp → wall_follow → 50k steps** → live progress, mean return, W&B curve, status `COMPLETE`.

See `docs/PHASES.md`, `docs/PHASE_1_APIS.md`, `docs/PHASE_1_TEST.md`.
