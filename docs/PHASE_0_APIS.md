# Phase 0 APIs (documented before code)

Exact tools and signatures RoboLab Phase 0 will use. Links are to current official docs. No Phase 1+ training/sim APIs here.

## Ports

| Service | Port | How started |
|---------|------|-------------|
| FastAPI (`robolab_api`) | `8000` | `uv run --package robolab-api uvicorn …` |
| Vite web dashboard | `5173` | `npm run dev` in `web/` |

`make dev` starts both concurrently.

---

## 1. uv workspace

Docs: [Using workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)

Root `pyproject.toml`:

```toml
[project]
name = "robolab-workspace"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.uv]
package = false

[tool.uv.workspace]
members = ["robolab", "robolab_eval", "robolab_api"]
```

Workspace member dependency (example: API depends on core):

```toml
[project]
dependencies = ["robolab", "fastapi", "uvicorn[standard]", "sqlmodel"]

[tool.uv.sources]
robolab = { workspace = true }
```

Commands:

| Command | Purpose |
|---------|---------|
| `uv sync` | Install workspace root env + lockfile |
| `uv lock` | Resolve entire workspace |
| `uv run --package robolab-api <cmd>` | Run command in member context |

Package names (PEP 621 `name`):

| Directory | `project.name` | Import path |
|-----------|----------------|-------------|
| `robolab/` | `robolab` | `robolab` |
| `robolab_eval/` | `robolab-eval` | `robolab_eval` |
| `robolab_api/` | `robolab-api` | `robolab_api` |

Build backend for library members: Hatchling (`[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"`).

---

## 2. FastAPI application

Docs: [First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/), [SQL Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/), [CORS](https://fastapi.tiangolo.com/tutorial/cors/)

```python
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title="RoboLab API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

`CORSMiddleware` kwargs (from FastAPI CORS docs): `allow_origins`, `allow_origin_regex`, `allow_methods`, `allow_headers`, `allow_credentials`, `expose_headers`, `max_age`.

Phase 0 HTTP routes:

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/health` | `{"status": "ok"}` |
| `GET` | `/api/experiments` | `{"experiments": [], "count": 0}` |
| `GET` | `/api/budget` | see Budget below |

---

## 3. SQLModel + SQLite

Docs: [Create DB and tables](https://sqlmodel.tiangolo.com/tutorial/create-db-and-table/), [Session with FastAPI dependency](https://sqlmodel.tiangolo.com/tutorial/fastapi/session-with-dependency/), [FastAPI SQL databases](https://fastapi.tiangolo.com/tutorial/sql-databases/)

```python
from sqlmodel import Field, Session, SQLModel, create_engine, select

class Experiment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str

sqlite_url = "sqlite:///./robolab.db"
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]
```

Signatures used:

| API | Signature / usage |
|-----|-------------------|
| `Field` | `Field(default=None, primary_key=True)` |
| `create_engine` | `create_engine(url: str, *, connect_args: dict = ..., echo: bool = False)` |
| `SQLModel.metadata.create_all` | `create_all(bind: Engine)` |
| `Session` | `with Session(engine) as session:` |
| `select` | `session.exec(select(Experiment)).all()` |

Phase 0: tables exist; list endpoint returns empty until Phase 1 creates rows. No seeded mock rows.

---

## 4. Uvicorn

Docs: [Settings](https://uvicorn.dev/settings/)

CLI:

```bash
uv run --package robolab-api uvicorn robolab_api.main:app --host 127.0.0.1 --port 8000 --reload --env-file .env
```

| Flag | Meaning |
|------|---------|
| `--host` | Bind host (default `127.0.0.1`) |
| `--port` | Bind port (default `8000`) |
| `--reload` | Dev auto-reload |
| `--env-file` | Load `.env` before app start (Uvicorn ≥ 0.21) |

Programmatic equivalent: `uvicorn.run("robolab_api.main:app", host="127.0.0.1", port=8000, reload=True)`.

---

## 5. Budget from `.env`

Phase 0 formula (month spend = `0`):

```text
remaining_usd = BUDGET_USD_CAP - month_spend_usd
```

Read via `os.environ` after Uvicorn `--env-file .env` (or `python-dotenv` `load_dotenv()` if needed).

```python
import os

def get_budget() -> dict:
    cap = float(os.environ.get("BUDGET_USD_CAP", "100"))
    month_spend = 0.0  # Phase 0: no cost ledger yet
    return {
        "budget_usd_cap": cap,
        "month_spend_usd": month_spend,
        "remaining_usd": cap - month_spend,
    }
```

`.env.example` keys (no secrets committed):

```bash
RUNPOD_API_KEY=
WANDB_API_KEY=
ANTHROPIC_API_KEY=
BUDGET_USD_CAP=100
```

---

## 6. Vite + React + TypeScript

Docs: [Vite Getting Started](https://vite.dev/guide/), [Create Vite](https://vite.dev/guide/#scaffolding-your-first-vite-project)

Scaffold:

```bash
npm create vite@latest web -- --template react-ts
```

`vite.config.ts` (proxy API to backend):

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
```

Vite `defineConfig` / `server.port` / `server.proxy`: [Server Options](https://vite.dev/config/server-options.html).

Frontend fetches:

```ts
const res = await fetch("/api/budget");
const data = await res.json();
// data.remaining_usd → header "Budget: $X remaining"

const exp = await fetch("/api/experiments");
const { count } = await exp.json();
// count === 0 → "0 experiments"
```

---

## 7. Tailwind CSS v4 (Vite plugin)

Docs: [Installing Tailwind CSS with Vite](https://tailwindcss.com/docs/installation/using-vite)

```bash
npm install tailwindcss @tailwindcss/vite
```

```ts
import tailwindcss from "@tailwindcss/vite";
// plugins: [react(), tailwindcss()]
```

CSS entry:

```css
@import "tailwindcss";
```

Recharts is allowed by the Phase 0 stack but **not required** for an empty experiments table; deferred until a chart is needed.

---

## 8. `make dev` concurrency

Makefile uses a simple shell background pattern (no extra npm dep required):

```makefile
dev:
	uv sync
	cd web && npm install
	uv run --package robolab-api uvicorn robolab_api.main:app --host 127.0.0.1 --port 8000 --reload --env-file .env & \
	cd web && npm run dev
```

Alternatively `npx concurrently` if we add it later (ask before adding). Phase 0 prefers Makefile-only concurrency.

---

## 9. Explicitly out of scope for Phase 0

MuJoCo, SB3, RunPod, W&B training, SSE, `POST /runs`, Recharts charts, seed data in experiments table.
