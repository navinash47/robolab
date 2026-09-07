# Phase: Architecture Builder + KAF/GPKAN/FAN (researched before code)

**Paper:** [arXiv:2502.06018](https://arxiv.org/abs/2502.06018) — *Kolmogorov-Arnold Fourier Networks* (Zhang et al.). DOI `10.48550/arXiv.2502.06018`.

**Scope:** Wall-follow only for training proof. Extend arch registry with **kaf / gpkan / fan**. New **Architectures** dashboard tab to author named arch configs and use them from **New Run**. RunPod-only for 100k proof (Secure EU-RO-1 + volume). Do not break `mlp` / `kan`.

**Doc wins** if it contradicts a later build prompt.

---

## 1. Paper → RoboLab name map (verified)

| UI / registry name | Paper role | What it is |
|---|---|---|
| **`kaf`** | **Main contribution** of 2502.06018 | **Kolmogorov-Arnold Fourier Network.** Replaces KAN B-splines with trainable **Random Fourier Features (RFF)** + hybrid **GELU–Fourier** activation with learnable scales `α` (base) / `β` (spectral). Official code: `kolmogorovArnoldFourierNetwork/KAF` (`FastKAFLayer` / `RandomFourierFeatures`). |
| **`fan`** | Baseline cited in §2 / §4 | **Fourier Analysis Network** (Dong et al., arXiv:2410.02675, NeurIPS’25). Layer: `[cos(W_p x) ‖ sin(W_p x) ‖ σ(B + W_g x)]` with **`p_ratio`** (paper default **0.25**). Code: `YihongDong/FAN` (`FANLayer`). |
| **`gpkan`** | Baseline cited as GPKAN (Yang & Wang 2024) in KAF experiments | **Gaussian-basis KAN-style** edge functions (GP / RBF spirit of [GP-KAN arXiv:2407.18397](https://arxiv.org/abs/2407.18397)). Full probabilistic GP neurons are too heavy for SB3 PPO; RoboLab implements a **deterministic RBF–KAN**: GELU base + learnable Gaussian RBF centers/widths/coeffs per edge, matching the paper’s “GELU-based initialization” experimental note. |
| `mlp`, `kan` | Existing | Unchanged. |

**Not in this phase:** Genesis/Isaac training matrices, Phase 6 transfer, full probabilistic GP uncertainty heads.

---

## 2. Architecture modules (PyTorch ↔ SB3)

Same ABC as Phase 1/2 (`robolab/core/arch.py`):

```python
@register_arch("kaf")   # also gpkan, fan
class …(Architecture):
    def forward(self, obs) -> tuple[pi_latent, vf_latent]: ...
```

Trainer already passes `policy_kwargs={"arch_name", "arch_cfg"}` into `RoboLabActorCriticPolicy` → `ArchExtractor`. **No ABC change.**

### Default `arch_cfg` (New Run builtins)

```yaml
# kaf — from paper / official FastKAFLayer defaults
hidden_sizes: [64, 64]
num_grids: 8                 # RFF frequency count (paper also explores “9 grids”)
activation_expectation: 1.64 # σ for RFF init (CLT / SiLU–GELU alignment)
use_layernorm: true
spline_dropout: 0.0
lr_default: 3.0e-4           # optional UI hint only; trainer uses TrainerCfg.lr

# gpkan
hidden_sizes: [32, 32]
num_basis: 8                 # Gaussian RBF centers per edge
init_bandwidth: 1.0
base_activation: gelu

# fan — Dong et al.
hidden_sizes: [64, 64]
p_ratio: 0.25                # must be in (0, 0.5)
activation: gelu
```

Files:

- `robolab/robolab/archs/kaf.py` — port of `RandomFourierFeatures` + `FastKAFLayer` + stack (no `print`, no Amp dependency).
- `robolab/robolab/archs/fan.py` — port of `FANLayer` stack.
- `robolab/robolab/archs/gpkan.py` — RBF–KAN stack as above.
- `robolab/robolab/archs/__init__.py` — import all five builtins.

---

## 3. Saved architectures (Builder → registry)

### Storage

1. **SQLite table `saved_arch`** (primary for dashboard CRUD).
2. **Optional mirror** `configs/arches/<slug>.yaml` on save (git-friendly; not required for train).

```python
class SavedArch(SQLModel, table=True):
    __tablename__ = "saved_arch"
    id: str                 # uuid hex[:12]
    name: str               # unique, user-editable slug (e.g. kaf_wall_narrow)
    base_arch: str          # kaf | gpkan | fan | mlp | kan
    cfg_json: str           # arch_cfg dict
    notes: str = ""
    created_at: datetime
    updated_at: datetime
```

### Resolution at run create

```
arch dropdown value:
  - builtin: "mlp" | "kan" | "kaf" | "gpkan" | "fan"
  - custom:  "custom:<saved_arch.id>"   OR bare saved name if unique

Trainer RunConfig.arch = base_arch (builtin name for registry).
RunConfig.arch_cfg = merged saved cfg (or builtin defaults).
Run.name / W&B tags may include saved name for UX.
```

`GET /api/archs` becomes:

```json
{
  "archs": ["fan", "gpkan", "kaf", "kan", "mlp"],
  "saved": [
    {"id": "a1b2c3d4e5f6", "name": "kaf_wall_narrow", "base_arch": "kaf", "cfg": {...}}
  ]
}
```

### REST

| Method | Path | Body / notes |
|---|---|---|
| `GET` | `/api/architectures` | List saved (+ builtins metadata) |
| `POST` | `/api/architectures` | `{name, base_arch, cfg, notes?}` → create |
| `GET` | `/api/architectures/{id}` | One saved |
| `PUT` | `/api/architectures/{id}` | Update name/cfg/notes |
| `DELETE` | `/api/architectures/{id}` | Delete |

Validation: `base_arch ∈ list_archs()`; `name` `[a-zA-Z0-9_\-]{1,64}` unique; cfg keys allowed per base (extra keys allowed but ignored by module).

---

## 4. Dashboard UI

- Nav tab **Architectures** next to Experiments / Compare (`data-testid="nav-architectures"`).
- Builder form: base type select → knobs for that type (layers as comma widths, num_grids / num_basis / p_ratio / layernorm / lr_default) → **Save architecture** (name + id shown).
- List of saved arches with delete.
- **New Run** arch dropdown: builtins + `saved.name (base)` entries; on select, POST uses `arch=base_arch` and `arch_cfg` from saved or `archCfgFor(builtin)`.
- Timestep presets: add **100k** (default for kaf/gpkan/fan when selected is optional UX; form may keep prior timesteps — prefer exposing `100000` in the select).

Match existing RoboLab look (`--accent`, surface cards, IBM Plex). No over-design.

---

## 5. Training defaults / RunPod

- Task: **`wall_follow`** only for gate proof (other tasks still allowed if already registered).
- Default train length for these arches: **100_000** steps (user runs; smoke **2k/5k** OK).
- Compute for proof: **runpod**, Secure + EU-RO-1 + volume from `.env`. No Mac Isaac.
- Same URDF `diffdrive_lidar` / wall_follow as existing.

---

## 6. Data flow (Builder → wall_follow)

```text
Architectures UI  →  POST /api/architectures  →  SQLite saved_arch
                                              ↘  optional configs/arches/*.yaml

New Run UI  →  POST /api/runs {arch: base, arch_cfg: …}
            →  local|runpod trainer
            →  get_arch(base)(obs, act, cfg)
            →  RoboLabActorCriticPolicy / PPO
            →  wall_follow env (mujoco|pybullet|…)
```

Mermaid (also in TEST / ARCH diagram doc after ship):

```mermaid
flowchart LR
  UI[Architectures tab] -->|POST /api/architectures| DB[(saved_arch SQLite)]
  UI2[New Run] -->|GET /api/archs| Reg[Builtin registry + saved]
  UI2 -->|POST /api/runs| API[runs router]
  API --> Train[trainer.train]
  Train --> Pol[RoboLabActorCriticPolicy]
  Pol --> Arch[get_arch kaf|gpkan|fan|mlp|kan]
  Train --> Env[wall_follow + URDF]
```

---

## 7. Non-goals / flags

- Do **not** vendor full KAF repo as a pip dep (license/print noise); **port** the layer into `robolab/archs/`.
- Do **not** change `Architecture.forward` return type.
- Do **not** commit `.env` / secrets.
- Dirty-tree gate: does not refuse; falls back to `origin/main` / pushed SHA (commit+push still needed for pod to see local edits).

---

## 8. Open questions (defaults chosen)

| Question | Default if unanswered |
|---|---|
| Full probabilistic GP-KAN vs RBF–KAN? | **RBF–KAN** (`gpkan`) for SB3 compatibility |
| Exact Yang & Wang 2024 citation mismatch vs Chen GP-KAN? | Documented; implement RBF–KAN labeled `gpkan` |
| Auto-select 100k when picking kaf/gpkan/fan? | Expose **100k** preset; do not forcibly overwrite user timesteps |
| Persist YAML always? | Write `configs/arches/<name>.yaml` on save; SQLite is source of truth for API |
