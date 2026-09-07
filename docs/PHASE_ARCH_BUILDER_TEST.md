# Phase: Architecture Builder + KAF/GPKAN/FAN — TEST notes

**Status (2026-09-07):** Code + API + Architectures tab shipped. Local SB3 smoke (`learn` 128 steps) **COMPLETE** for `kaf` / `gpkan` / `fan` on `wall_follow` + mujoco. **100k RunPod proof** ready for user matrix (dirty-tree gate no longer blocks; falls back to remote SHA). Human UI gate awaiting confirm.

## Paper map (verified)

| Registry | Meaning |
|---|---|
| `kaf` | Kolmogorov-Arnold Fourier (main paper arXiv:2502.06018) — RFF + GELU hybrid |
| `fan` | Fourier Analysis Network (Dong et al.) — cos/sin + σ path, `p_ratio=0.25` |
| `gpkan` | Gaussian RBF KAN-style baseline (GP-KAN spirit; deterministic for SB3) |

## Automated / CLI checks done

```bash
# Registry + forward
uv run python -c "import robolab.archs; from robolab.core.arch import list_archs; print(list_archs())"
# → fan, gpkan, kaf, kan, mlp

# SB3 short learn (mujoco wall_follow) — kaf/gpkan/fan each 128 steps OK

# API (after launchctl kickstart …/com.robolab.api8000)
curl -s http://127.0.0.1:8000/api/archs          # builtins + defaults + saved
curl -s -X POST http://127.0.0.1:8000/api/architectures \
  -H 'Content-Type: application/json' \
  -d '{"name":"kaf_wall_smoke","base_arch":"kaf","cfg":{"hidden_sizes":[32,32],"num_grids":6}}'
```

`web` `npm run build` (tsc + vite) OK.

## Exact UI clicks — Builder → New Run

1. Open `http://localhost:5173`
2. Tab **Architectures**
3. Base **kaf** (or gpkan/fan), tweak knobs, name e.g. `kaf_wall_narrow`
4. **Save architecture** → row appears with id
5. **Use** / **New Run** → Arch dropdown shows `name (base)` under Saved
6. Compute **runpod**, Task **wall_follow**, Timesteps **100k**, Start  
   (or **2k/5k** smoke first)

## Architecture data flow

```text
Architectures UI
    │ POST /api/architectures
    ▼
SQLite saved_arch  ──mirror──► configs/arches/<name>.yaml
    │
New Run (arch=custom:<id> | builtin)
    │ POST /api/runs  → resolve → base_arch + arch_cfg
    ▼
trainer.train → RoboLabActorCriticPolicy → get_arch(base)
    ▼
wall_follow + same URDF (mujoco|pybullet|…)
```

See `docs/PHASE_ARCH_BUILDER_DIAGRAM.md` for mermaid.

## Human gate checklist

- [ ] Architectures tab visible next to Experiments / Compare
- [ ] Save named arch; appears in New Run dropdown
- [ ] Builtin dropdown lists `mlp`, `kan`, `kaf`, `gpkan`, `fan`
- [ ] Optional: RunPod 100k wall_follow for each of kaf/gpkan/fan (Secure EU-RO-1)
- [ ] Compare curves after COMPLETE

## Non-blocking notes

- `gpkan` is **RBF–KAN**, not full probabilistic GP-KAN (documented in APIs).
- Smoke saved arch `kaf_wall_smoke` may exist in local SQLite from API test.
