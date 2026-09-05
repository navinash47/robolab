# Isaac Lab / Isaac Sim — install reality check

**Host used for this note:** macOS Apple Silicon (Darwin / M-series), **no NVIDIA GPU**.

## Blunt verdict

| Platform | Isaac Sim | Isaac Lab | RoboLab status |
|---|---|---|---|
| **macOS (this laptop)** | **Not supported** | **Not supported** | Stubs only — do **not** pretend it works |
| Linux x86_64 + NVIDIA RTX (≥16 GB VRAM recommended) | Supported | Supported (on top of Sim) | Real path = **RunPod Secure** (or any Linux+NVIDIA box) |
| Windows 11 + NVIDIA | Supported | Supported | Out of scope for RoboLab workers |

Official requirements (checked 2026-09): [Isaac Lab local install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and [Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) list **Ubuntu 22.04/24.04 or Windows 11**, **≥32 GB RAM**, **≥16 GB GPU VRAM**, NVIDIA production drivers. **macOS is not an OS target.** Pip install is `isaacsim[all,extscache]==5.1.0` from `https://pypi.nvidia.com` (Python **3.11**, GLIBC 2.35+). Containers are **Linux-only**.

## What RoboLab does on Mac

1. Keep `isaaclab` / `isaac_sim` **stubs** in the sim registry (actionable `RuntimeError` + `requires_nvidia`).
2. **Never** run wall_follow / installs / smoke tests for Isaac or Genesis on the Mac when the directive is RunPod-only.
3. Train on **RunPod Secure** in **EU-RO-1** with network volume `1hyuaan8i2` (`RUNPOD_CLOUD_TYPE=SECURE`).

## RunPod path (cost-aware)

### Already in the worker image (`avinashnandyala2/robolab-worker:phase3`)

- MuJoCo + PyBullet + SB3 + torch CUDA base — **ready for wall_follow on RunPod**.

### Genesis (lightweight add-on)

- Package: `genesis-world` (~80 MB wheel + deps).
- Installed **at pod start** when `sim=genesis` (cached under `/workspace/.cache` on the network volume) — **no multi-GB image rebuild required**.
- Prefer GPU: set `ROBOLAB_GENESIS_GPU=1` in the pod env (entrypoint / create path).

### Isaac Lab / Sim (HEAVY — human gate)

Baking Isaac into the worker is a **multi‑GB** (often **15–50+ GB**) NVIDIA Omniverse stack: long build, large registry push, slower cold start, and RT-core GPUs only (A100/H100 **not** suitable for Isaac Sim rendering).

**Do not rebuild the default `phase3` worker with Isaac unless explicitly approved.** Options when approved:

1. **Separate image** e.g. `…/robolab-worker:isaac` FROM an NVIDIA Isaac Sim container + RoboLab bootstrap (keep `phase3` lean for mujoco/pybullet/genesis).
2. **One-shot pod install** onto the 40 GB volume (tight — may need a larger volume).
3. Point `ROBOLAB_WORKER_IMAGE` at the Isaac-tagged image only for Isaac runs.

Pip sketch (Linux GPU pod / image, **not** Mac):

```bash
# Python 3.11 venv on Linux + NVIDIA
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# then clone Isaac Lab and follow:
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
```

Until a real Gymnasium adapter ships, registry names still **fail loudly** with install + RunPod pointers even if packages are present.

## Spend sketch (Secure EU-RO-1)

| GPU (Secure) | ~$/hr (catalog) | Notes |
|---|---|---|
| RTX 4090 | ~0.74 | Default preference; High stock in EU-RO-1 |
| RTX 3090 | ~0.22–0.40 | Cheaper fallback when stocked |
| A40 | ~0.35 | Secure High stock — **no RT cores → bad for Isaac Sim** |

Phased wall_follow matrix (sequential, self-terminate): 2k → 5k → one 50k → one ~100k per **working** sim. Mujoco+PyBullet only is typically **well under $5** if pods terminate. Full Genesis add-on is still usually **<$5**. A full Isaac image build+push+smoke can **exceed $5** — **ask first**.

## Human input needed

1. **Approve Isaac worker image** (`:isaac`, multi‑GB) before any bake/push.
2. Keep **cloudflared** (or equivalent) alive so `BACKEND_PUBLIC_URL` reaches pods.
3. Optional: raise volume size if installing Isaac onto `1hyuaan8i2` (40 GB).

## Links

- [Isaac Lab install index](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Sim pip package](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
- [Isaac Sim system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [Genesis install (Mac/Linux/Windows; CPU OK)](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)
- RoboLab: `docs/SIMULATORS.md`, `docs/PHASE_3_SETUP.md`
