# Phase 6 APIs (researched before code)

Sim-to-sim zero-shot transfer after training. **Doc wins** if it contradicts a build prompt.

Sources:

- `docs/PHASES.md` Phase 6 gate (transfer_to → Transfer report)
- Existing `RunConfig.transfer_to`, `DomainParams` (friction / mass_scale / sensor_noise_std)
- Gate sims: **mujoco + pybullet** (same URDF). Genesis best-effort. Isaac = Mac stubs / RunPod `:isaac` only
- Phase 4 checkpoint load path (`policy.zip`, PPO / tabular Q / FA Q)

## Conflict table

| Assumption | Reality | What we do |
|---|---|---|
| Train then auto-eval in `transfer_to` sims | Trainer already posts `/complete` after checkpoint; no transfer hook yet | After checkpoint save, if `transfer_to` non-empty, run transfer eval **before** `/complete`; also support post-hoc `POST /api/runs/{id}/transfer` |
| `mass_scale` changes dynamics | Gate sims are **kinematic** planar drive; mass often unused | Still sweep `mass_scale` on the domain object; document curves may be flat until full dynamics |
| Isaac/Genesis transfer on Mac | Unsupported / best-effort | Skip with `status: skipped` + reason; never Mac train+render Isaac/Genesis |
| Transfer ratio for negative returns | wall_follow returns can be negative | `gap = source − target`; `transfer_ratio = target / source` only when `\|source\| ≥ 1e-6`, else `null` |

## Config

`RunConfig.transfer_to: list[str] | null` (already on the model).

Example New Run / YAML:

```yaml
sim: mujoco
transfer_to: [pybullet]
```

UI: New Run multi-select **Transfer to** (gate sims except the train sim). Default empty → no auto transfer (unchanged Phase 1–5 behavior).

## Transfer eval (library)

Module: `robolab.train.transfer`

```python
def run_transfer_eval(
    *,
    cfg: RunConfig,
    checkpoint: Path,
    run_id: str,
    targets: list[str] | None = None,   # default: cfg.transfer_to
    n_episodes: int = 5,
    seed: int = 0,
    out_dir: Path | None = None,       # default transfer/{run_id}
) -> dict:
    ...
```

### Procedure (per target)

1. Resolve policy from `checkpoint` (same loaders as Phase 4 render: PPO / `TabularQAgent` / `FunctionApproxQAgent`).
2. **Source baseline:** `n_episodes` deterministic rollouts in `cfg.sim` with `cfg.domain` (train domain).
3. **Zero-shot target:** same policy, same task/robot/domain, env from each `target` sim.
4. **Perturbation robustness** (source + each successful target):
   - `friction`: `[0.5, 1.0, 1.5, 2.0]`
   - `mass_scale`: `[0.75, 1.0, 1.25]`
   - `sensor_noise_std`: `[0.0, 0.02, 0.05]`
   - One episode mean-return per level (fast); keep axis values in report for curves.
5. Skip unknown / unavailable sims (`genesis` without install, Isaac on Mac) → entry with `status: "skipped"`.

### Metrics

```text
source.mean, source.std, source.ci_low, source.ci_high, source.returns[]
targets[].sim, targets[].status, targets[].mean, …, targets[].transfer_ratio, targets[].gap
robustness.{friction|mass_scale|sensor_noise_std}[].{sim, value, mean_return}
```

CI: normal approx `mean ± 1.96 * std / sqrt(n)` when `n ≥ 2`, else `ci = null`.

W&B (best-effort when a run is active): log `transfer/source_mean`, `transfer/{sim}_mean`, `transfer/{sim}_ratio`, `transfer/{sim}_gap`.

Artifacts on disk:

```text
transfer/{run_id}/report.json
```

## HTTP API

### `GET /api/runs/{run_id}/transfer`

Returns the report JSON (from DB cache or `transfer/{run_id}/report.json`).

- `404` if no report yet.
- Response includes `run_id`, `source_sim`, `targets`, `robustness`, `status` (`READY` | `FAILED` | `RUNNING`).

### `POST /api/runs/{run_id}/transfer`

Queue **local** transfer eval for a `COMPLETE` run with a local `policy.zip` (or resolvable checkpoint).

Body (all optional):

```json
{
  "targets": ["pybullet"],
  "n_episodes": 5
}
```

- Default `targets` = `config.transfer_to` or `[pybullet]` if train sim is `mujoco`, else `[mujoco]`.
- Sets `transfer_status=RUNNING`, spawns subprocess (same pattern as local render), then `READY` / `FAILED`.
- Does **not** start RunPod or long trains.
- Isaac/Genesis targets may be skipped locally.

### Run list / detail fields (additive)

```json
{
  "transfer_status": "READY" | "RUNNING" | "FAILED" | null,
  "transfer_summary": {
    "source_mean": 12.3,
    "targets": [{"sim": "pybullet", "mean": 10.1, "transfer_ratio": 0.82, "gap": 2.2}]
  } | null
}
```

### Complete payload (optional)

Trainer may include `transfer` object in `/complete` body; API persists report + sets `transfer_status=READY`.

## UI

- Nav tab **Transfer**.
- Pick a COMPLETE run → **Run transfer** (if no report) or show report.
- Table: source vs each target (mean, CI, ratio, gap).
- One chart per perturbation axis (friction / mass / noise) with a series per sim.

## Non-goals (Phase 6)

- Seed fan-out / `rliable` (Phase 7)
- Matrix orchestration (Phase 8)
- Mac Genesis/Isaac train+render
- Auto-start 2M-step or overnight pods
- Changing Abort, token tracker, Arch Builder, or Phase 0–5 gate semantics
