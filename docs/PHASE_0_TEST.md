# Phase 0 human gate — verification script

**Definition of done:** run `make dev`, open `localhost:5173`, see RoboLab with **"0 experiments"** and **"Budget: $X remaining"**.

Blunt status: Phase 0 is complete when every PASS item below succeeds. Anything unchecked is incomplete — do not start Phase 1.

---

## Prerequisites

1. Repo: `/Users/avinashnandyala/Projects/robolab`
2. Tools installed: `uv`, `node`/`npm`, `make`
3. Optional: edit `.env` (copied from `.env.example` by `make dev` if missing). Set `BUDGET_USD_CAP` (default `100`).

---

## Exact steps

### 1. Start the stack

```bash
cd /Users/avinashnandyala/Projects/robolab
make dev
```

**Expect in terminal:**

- `uv sync` finishes without error
- `npm install` in `web/` finishes
- Uvicorn line roughly: `Uvicorn running on http://127.0.0.1:8000`
- Vite line roughly: `Local: http://localhost:5173/`

**Fail looks like:**

- `uv: command not found` → install uv
- Python version error → Phase 0 requires Python ≥ 3.11 (`uv` installs it)
- Port already in use on 8000 or 5173 → kill the other process and retry
- Vite starts but API never binds → backend failed; scroll for Python traceback

### 2. API smoke (optional but recommended)

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api/budget
curl -s http://127.0.0.1:8000/api/experiments
```

**PASS:**

```json
{"status":"ok"}
{"budget_usd_cap":100.0,"month_spend_usd":0.0,"remaining_usd":100.0}
{"experiments":[],"count":0}
```

(`budget_usd_cap` / `remaining_usd` match your `BUDGET_USD_CAP`.)

**Fail:** connection refused, 500, non-empty `experiments`, or `count` ≠ 0.

### 3. Open the dashboard (required gate)

1. Open a browser to **http://localhost:5173**
2. Page title / brand: **RoboLab**
3. Header shows text matching: **Budget: $100 remaining** (or your cap; currency formatting may show `$100.00`)
4. Experiments area shows **0 experiments** (empty table body — not fake rows)

**PASS checklist (copy/paste, tick as you go):**

```
[ ] make dev starts without crashing
[ ] http://127.0.0.1:8000/health returns ok
[ ] http://localhost:5173 loads
[ ] Page shows brand "RoboLab"
[ ] Header shows "Budget: $X remaining" (X from BUDGET_USD_CAP)
[ ] Page shows "0 experiments" (empty state, no mock rows)
```

**Fail looks like:**

- Blank page / Vite error overlay → JS build issue
- Red banner "Failed to reach API" → backend down or proxy broken
- Budget stuck on "Budget: …" → `/api/budget` failed
- Any invented experiment rows → Phase 0 violated; wipe DB / fix seed
- Wrong port (not 5173) → check Vite output

### 4. Budget env check (optional)

```bash
# stop make dev (Ctrl+C), then:
echo 'BUDGET_USD_CAP=42' > .env
# restore other keys from .env.example if needed
make dev
```

Reload http://localhost:5173 → header should show **Budget: $42 remaining** (or `$42.00`).

### 5. Stop

Ctrl+C in the `make dev` terminal. Both API and Vite should stop (`trap` kills the process group).

---

## Incomplete / deferred (expected)

- No MuJoCo / SB3 / RunPod / W&B / training / `POST /runs`
- `docker/worker.Dockerfile` is a stub (`make worker-image` only prints)
- No automated unit tests (`make test` points here)
- Recharts not installed (empty table does not need charts)
- Cost ledger / month spend always `0` until Phase 3

If any required PASS item fails, Phase 0 is **not done**.
