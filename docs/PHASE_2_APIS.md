# Phase 2 APIs (researched before code)

Exact APIs and signatures for Phase 2. **Doc wins** if it contradicts the build prompt — flagged below.

## Choice: KAN library = **Blealtan/efficient-kan** (not pykan)

| Option | Install / Mac CPU notes | Decision |
|---|---|---|
| **[Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)** | Pure PyTorch; `KAN` / `KANLinear`; `pip/uv` from GitHub | **Selected** — installs cleanly, no sympy/conda stack |
| [KindXiaoming/pykan](https://github.com/KindXiaoming/pykan) | Docs push conda + `pip install git+…`; heavier / slower original impl | Rejected for local CPU smoke |
| [shawcharles/efficient-kan](https://github.com/shawcharles/efficient-kan) | Maintained fork of Blealtan API | Fallback if Blealtan install fails |

**Install (uv / pyproject):**

```toml
# robolab/pyproject.toml
dependencies = [
  # ...
  "efficient-kan @ git+https://github.com/Blealtan/efficient-kan.git",
]
```

**Package metadata** ([pyproject.toml](https://github.com/Blealtan/efficient-kan/blob/master/pyproject.toml)): name `efficient-kan`, `requires-python >=3.8`, deps include `torch>=2.3.0` (already in stack). Import path: `efficient_kan`.

### Signatures used

**Docs / source:** [kan.py](https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py), [`__init__.py`](https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/__init__.py)

```python
from efficient_kan import KAN, KANLinear
import torch

# Single layer
layer = KANLinear(
    in_features: int,
    out_features: int,
    grid_size: int = 5,
    spline_order: int = 3,
    scale_noise: float = 0.1,
    scale_base: float = 1.0,
    scale_spline: float = 1.0,
    enable_standalone_scale_spline: bool = True,
    base_activation=torch.nn.SiLU,
    grid_eps: float = 0.02,
    grid_range: list = [-1, 1],
)

# Stack: layers_hidden = [in, h1, h2, ..., out]
model = KAN(
    layers_hidden: list[int],  # e.g. [obs_dim, 32, 32, latent_dim]
    grid_size: int = 5,
    spline_order: int = 3,
    scale_noise: float = 0.1,
    scale_base: float = 1.0,
    scale_spline: float = 1.0,
    base_activation=torch.nn.SiLU,
    grid_eps: float = 0.02,
    grid_range: list = [-1, 1],
)

y = model(x)                      # x: (batch, in_features) → (batch, out_features)
y = model(x, update_grid=False)   # update_grid=True mutates spline grids (training-only; we leave False)
loss_reg = model.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0)
```

**RoboLab wrapper** (`robolab/archs/kan.py`) — same ABC as MLP; no core ABC break:

```python
from robolab.core.arch import Architecture, register_arch

@register_arch("kan")
class KANPolicyNet(Architecture):
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None): ...
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pi_features, vf_features)."""
    # param_count() inherited: sum(p.numel() for p in self.parameters() if p.requires_grad)
```

Default `arch_cfg` for New Run when arch=`kan`:

```python
{
  "hidden_sizes": [32, 32],
  "grid_size": 5,
  "spline_order": 3,
  "activation": "silu",  # informational; efficient-kan uses SiLU base
}
```

Smaller hidden than MLP `[64,64]` — KAN splines are heavier on CPU.

## Phase 1 architecture registry (extend, do not break)

**Existing** (`robolab/core/arch.py`):

```python
def register_arch(name: str): ...
def get_arch(name: str) -> type[Architecture]: ...
def list_archs() -> list[str]: ...

class Architecture(nn.Module, ABC):
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None): ...
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def param_count(self) -> int: ...
    @property
    def latent_dim_pi(self) -> int: ...
    @property
    def latent_dim_vf(self) -> int: ...
```

**Extend only:**

- Add `robolab/archs/kan.py` + import in `robolab/archs/__init__.py`
- Optional: `GET /api/archs` → `{"archs": list_archs()}` for the dropdown
- Trainer already passes `arch_name` / `arch_cfg` into `RoboLabActorCriticPolicy` — no ABC change required

**Flag:** Do **not** change `Architecture.forward` return type or `register_arch` semantics.

## Weights & Biases Public API — history for Compare

**Docs:** [Run (public API)](https://docs.wandb.ai/models/ref/python/public-api/run), [Export data / public API guide](https://docs.wandb.ai/models/track/public-api-guide)

Auth: `WANDB_API_KEY` in env (already used by Phase 1 trainer). `wandb.Api()` picks it up.

```python
import wandb

api = wandb.Api()
# path: "entity/project/run_id" — parse from run.wandb_url
# e.g. https://wandb.ai/avinashnandyala2-umass-amherst/robolab/runs/bvwqbhg7
run = api.run(f"{entity}/{project}/{wandb_run_id}")

# Sampled history (fast enough for overlay charts)
# Prefer pandas=False → list[dict] (avoids optional pandas; wandb may warn
# "Unable to load pandas" and still return a list when pandas=True).
rows = run.history(
    samples=500,                          # default 500
    keys=["rollout/ep_rew_mean"],         # Phase 1 logged key
    x_axis="_step",
    pandas=False,
)
# each row: {"_step": int, "rollout/ep_rew_mean": float, ...}

# Full scan (slower; use if samples drop points)
for row in run.scan_history(keys=["rollout/ep_rew_mean"], page_size=1000):
    step = row.get("_step")
    y = row.get("rollout/ep_rew_mean")

# Metadata for compare table
param_count = run.config.get("param_count") or run.summary.get("param_count")
arch = run.config.get("arch")
```

**Caveat (docs + community):** `scan_history(keys=[...])` only returns rows where **all** listed keys are present. Prefer a **single** metric key per call. Prefer `history(samples=…)` for the UI overlay.

**Phase 1 metric key:** `rollout/ep_rew_mean` (logged in `WandbAndHeartbeatCallback`).

**Phase 2 addition:** trainer logs `param_count` into `wandb_run.config` / `summary` after policy build so Compare can show counts for new runs. Fallback: instantiate arch from DB `config_json` and call `param_count()` (needs `obs_dim` from env/task — use known wall_follow lidar dim or store `param_count` on the SQLite `Run` row).

## Compare backend API (new)

```python
# POST /api/compare
class CompareRequest(BaseModel):
    run_ids: list[str]  # RoboLab run IDs (SQLite), length >= 1

class HistoryPoint(BaseModel):
    step: int
    mean_return: float | None

class CompareRun(BaseModel):
    id: str
    name: str
    arch: str
    status: str
    param_count: int | None
    mean_return: float | None
    wandb_url: str | None
    wandb_run_id: str | None
    history: list[HistoryPoint]
    error: str | None = None  # per-run fetch failure; others still returned

class CompareResponse(BaseModel):
    metric_key: str = "rollout/ep_rew_mean"
    runs: list[CompareRun]
```

Implementation sketch:

```python
from urllib.parse import urlparse

def parse_wandb_path(wandb_url: str) -> tuple[str, str, str]:
    # /{entity}/{project}/runs/{run_id}
    parts = urlparse(wandb_url).path.strip("/").split("/")
    entity, project, _, run_id = parts[0], parts[1], parts[2], parts[3]
    return entity, project, run_id

def fetch_history(wandb_url: str) -> list[dict]:
    entity, project, rid = parse_wandb_path(wandb_url)
    api = wandb.Api()
    wrun = api.run(f"{entity}/{project}/{rid}")
    rows = wrun.history(samples=500, keys=["rollout/ep_rew_mean"], pandas=False)
    ...
```

**No mock curves** — if W&B fetch fails, return `error` on that run; never invent points.

## Frontend (Compare + arch dropdown)

**Stack:** React 19 + Vite (existing). Chart: **Recharts** (build prompt).

```bash
# web/
npm install recharts
```

```tsx
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

// Merge per-run histories onto shared step axis for overlay, e.g.
// [{ step: 2048, "mlp-23421c": 12.3, "kan-abc": 8.1 }, ...]
```

UI:

- Nav / button **Compare**
- Multi-select COMPLETE (or any with `wandb_url`) runs
- Chart: one series per selected run (`rollout/ep_rew_mean` vs step)
- Table: name, arch, param_count, final mean_return, W&B link

New Run form: Arch `<select>` options from `GET /api/archs` or hardcode `mlp` | `kan` (must include both).

Timesteps: keep `50k` / `5k` / `2k`. Recommend **5k smoke** for local KAN CPU in TEST.md; 50k remains available.

## SQLite `Run` optional field

```python
class Run(SQLModel, table=True):
    # ... existing ...
    param_count: Optional[int] = None
```

Migrate with `ALTER TABLE run ADD COLUMN param_count INTEGER` if missing (SQLite `create_all` does not add columns).

Trainer / complete payload sets `param_count` from `model.policy.mlp_extractor.arch.param_count()`.

## Dependencies this phase

| Package | Where | Ask? |
|---|---|---|
| `efficient-kan` (git Blealtan) | `robolab` | Allowed by Phase 2 build prompt (KAN lib) |
| `recharts` | `web` | Required by Compare UI in build prompt |

No RunPod deps. No changes to Section 3 sim/train stack beyond KAN.

## Sources

- [Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan)
- [efficient-kan kan.py](https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py)
- [pykan (rejected)](https://github.com/KindXiaoming/pykan)
- [wandb Run.history / scan_history](https://docs.wandb.ai/models/ref/python/public-api/run)
- [wandb public API guide](https://docs.wandb.ai/models/track/public-api-guide)
- Phase 1: `robolab/core/arch.py`, `robolab/archs/mlp.py`, `WandbAndHeartbeatCallback` metric `rollout/ep_rew_mean`
