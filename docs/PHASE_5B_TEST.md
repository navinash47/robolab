# Phase 5B — quick TEST notes

## Gates that must work

- [ ] `GET /api/tasks` lists `docking`, `figure8_tracking`, `go_to_goal`, `maze`, `wall_follow`
- [ ] `GET /api/sims` lists `genesis`, `isaac_sim`, `isaaclab`, `mujoco`, `pybullet` with `meta.capabilities`
- [ ] New Run: Custom timesteps (e.g. `3333`) accepted; `0` / `10000001` rejected
- [ ] Local smoke: `mujoco` + `go_to_goal` + 2048 steps starts
- [ ] Local smoke: `pybullet` + `maze` + 2048 steps starts
- [ ] `isaaclab` / `isaac_sim` launch fails with install/stub message (not a silent hang)
- [ ] `genesis` without `genesis-world`: same actionable install error
- [ ] Experiment failure still flaggable; logistics auto-log unchanged

## Blunt coverage

| Sim | wall_follow + new tasks |
|---|---|
| mujoco | **Gate** — real |
| pybullet | **Gate** — real |
| genesis | Real adapter **if** `pip install genesis-world`; else install error. Treat as best-effort |
| isaaclab / isaac_sim | Stubs only |

See `docs/SIMULATORS.md`, `docs/PHASE_5B_APIS.md`, `docs/TASK_CANDIDATES.md`.
