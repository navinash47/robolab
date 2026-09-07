# Architecture Builder → wall_follow (diagram)

How a named architecture reaches PPO training on `wall_follow`.

## Mermaid

```mermaid
flowchart TB
  subgraph UI["Dashboard :5173"]
    ArchTab["Architectures tab\n(knobs + Save)"]
    NewRun["Experiments → New Run\n(arch dropdown)"]
  end

  subgraph API[":8000 FastAPI"]
    CRUD["POST/GET/PUT/DELETE\n/api/architectures"]
    Archs["GET /api/archs\nbuiltins + saved"]
    Runs["POST /api/runs\nresolve_arch_for_run"]
  end

  subgraph Store["Persistence"]
    DB[("SQLite saved_arch")]
    YAML["configs/arches/*.yaml\n(mirror)"]
  end

  subgraph Train["Trainer / worker"]
    Reg["Builtin registry\nmlp kan kaf gpkan fan"]
    Pol["RoboLabActorCriticPolicy\n+ ArchExtractor"]
    Env["wall_follow + URDF\nmujoco | pybullet | …"]
    PPO["SB3 PPO.learn"]
  end

  ArchTab --> CRUD
  CRUD --> DB
  CRUD --> YAML
  NewRun --> Archs
  Archs --> DB
  Archs --> Reg
  NewRun --> Runs
  Runs --> DB
  Runs --> Train
  Train --> Reg
  Reg --> Pol
  Pol --> PPO
  Env --> PPO
```

## ASCII

```
[Architectures UI] --POST--> [saved_arch SQLite] --yaml--> configs/arches/
         |                            ^
         v                            |
[New Run dropdown] <---- GET /api/archs
         |
         v
[POST /api/runs] --resolve--> arch=base_arch, arch_cfg=…
         |
         v
[trainer] -> get_arch(base) -> RoboLabActorCriticPolicy
         |
         v
[wall_follow env] <-> PPO.learn (local | RunPod)
```

## Builtin paper arches

| Name | Module | Source |
|---|---|---|
| `kaf` | `robolab/archs/kaf.py` | arXiv:2502.06018 (main) |
| `fan` | `robolab/archs/fan.py` | Dong et al. arXiv:2410.02675 |
| `gpkan` | `robolab/archs/gpkan.py` | RBF–KAN for GPKAN baseline |
