# Architecture rewards & hyperparameters (wall_follow)

Source of truth for **builtins** (`BUILTIN_DEFAULTS` in `robolab_api/.../architectures.py`)
and saved YAML under `configs/arches/`. Trainer defaults from `TrainerCfg`.

## Why training felt broken (dissection)

| Issue | Effect | Fix (this change) |
|-------|--------|-------------------|
| PPO `wall_follow` **TARGET_DIST = 0.35 m** | Policy hugs wall in PDF **Near** bin (≤0.7). PDF / Q-learning want **Medium 0.7–0.9 m**. PPO and QL fight each other. | Target **0.80 m** (medium center) |
| Camera pitch −35°, distance 3.5 | Side view of a tiny chassis — hard to see maze / RoboMaster motion | **Top-down** camera, longer distance |
| Chassis 0.30×0.24 m | Tiny vs course RoboMaster-scale arena | Larger **RoboMaster-style** top silhouette |
| Scale 2.5 → ~20 m | OK but not “very large” | Scale **4.0** → ~**32 m** outer span |
| Only combined largemaze | PDF Fig. 4 had **5** wall tests | Tasks: straight / L-in / L-out / I / U-turn |

**Retrain required** after this change — old checkpoints are for the old reward + maze.

---

## Task reward: `wall_follow` (PPO / continuous)

Used by env `task.reward` for **PPO/SAC**. Q-learning trainers **ignore** this and use PDF `pdf_reward` instead.

| Term | Value |
|------|--------|
| Target right standoff | **0.80 m** |
| Wall term | `1.2 * exp(-(err/0.25)²)` |
| Forward term | `0.8 * clip(v/0.6, −0.5, 1)` |
| Front close | −2 if front < 0.35 m |
| Crash | −5 |
| Stall/scrape | −3 if min_range < 0.25 and \|v\| < 0.08 |
| Success | \|right−0.80\| < 0.12 and v > 0.15 |
| Max steps | 4000 (large maze) |

## PDF reward: Q-learning (`avinash_wall` + Algo=Q-learning)

Exact P2_D3 piecewise R(s,a):

| Condition | R |
|-----------|---|
| Front near + turn left | **+15** |
| Front near otherwise | **−8** |
| Right = medium | **+20** |
| Right = far | **−5** |
| Right = near | **−1** |

Bins: Near ≤0.7, Medium ≤0.9, Far >0.9. Actions: left / forward / right at v=0.3, ω=±0.7.

---

## Builtin architectures

### `mlp` (PPO default)

| Hyper | Default |
|-------|---------|
| hidden_sizes | [64, 64] |
| activation | tanh |
| trainer.algo | ppo |
| lr / γ / n_steps / batch | 3e-4 / 0.99 / 2048 / 64 |

### `kan`

| Hyper | Default |
|-------|---------|
| hidden_sizes | [32, 32] |
| grid_size / spline_order | 5 / 3 |
| trainer (PPO) | same as mlp |

### `kaf` (+ `configs/arches/kaf_wall_follow.yaml`)

| Hyper | Default |
|-------|---------|
| hidden_sizes | [64, 64] |
| num_grids | 8 |
| activation_expectation | 1.64 |
| use_layernorm | true |
| spline_dropout | 0 |
| lr_default | 3e-4 |

### `gpkan`

| Hyper | Default |
|-------|---------|
| hidden_sizes | [32, 32] |
| num_basis / init_bandwidth | 8 / 1.0 |
| base_activation | gelu |
| lr_default | 3e-4 |

### `fan`

| Hyper | Default |
|-------|---------|
| hidden_sizes | [64, 64] |
| p_ratio | 0.25 |
| activation | gelu |
| lr_default | 3e-4 |

### `avinash_wall` (tabular Q)

| Hyper | Default |
|-------|---------|
| algorithm | q_learning |
| α / γ | 0.1 / **1.0** |
| ε | 1.0 → 0.05 over **total_timesteps** |
| linear_vel / angular_vel | 0.3 / 0.7 |
| near_max / medium_max | 0.7 / 0.9 |
| episode_max_steps | 1200 |
| lr_default | 0.1 |

### FA Q-learning (mlp/kan/kaf/gpkan/fan + Algo=Q-learning)

Same PDF actions/reward; Adam `trainer.lr` (default 3e-4); `obs_mode` lidar or onehot27; γ usually **1.0** in arch_cfg.

---

## Wall scenarios (P2_D3 Fig. 4)

| Task id | Scenario |
|---------|----------|
| `wall_follow` | Full scaled largemaze (all features) |
| `wall_straight` | Parallel corridor |
| `wall_l_inside` | Inside L-corner |
| `wall_l_outside` | Outside L-corner |
| `wall_i_corner` | I / T junction stub |
| `wall_uturn` | 180° U-turn channel |

Same reward / obs / action as `wall_follow`. Train on `wall_follow`, smoke each scenario, or curriculum.
