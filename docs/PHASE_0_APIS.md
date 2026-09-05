# Phase 0 APIs (researched before code)

Exact APIs and signatures planned for Phase 0. Doc wins over this prompt if they diverge.

## uv workspace

**Docs:** [Using workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/), [Working on projects](https://docs.astral.sh/uv/guides/projects/), [Managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/)

Root `pyproject.toml`:

```toml
[project]
name = "robolab-workspace"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.uv.workspace]
members = ["robolab", "robolab_eval", "robolab_api"]

[tool.uv.sources]
robolab = { workspace = true }
robolab-eval = { workspace = true }
robolab-api = { workspace = true }
```

Member packages use their own `pyproject.toml` with `[build-system]` (hatchling or uv_build). Workspace member deps via `{ workspace = true }` are editable.

Commands:

| Command | Purpose |
|---|---|
| `uv sync --all-packages` | Install all workspace members into `.venv` |
| `uv lock` | Resolve entire workspace lockfile |
| `uv run --package robolab-api …` | Run a command with that member’s env |
| `uv run --env-file .env …` | Load `.env` into the process env ([CLI](https://docs.astral.sh/uv/reference/cli/)) |

## FastAPI

**Docs:** [First steps](https://fastapi.tiangolo.com/tutorial/first-steps/), [CORS](https://fastapi.tiangolo.com/tutorial/cors/), [Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)

```python
from typing import Annotated
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RoboLab API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

Path ops for Phase 0:

- `GET /api/experiments` → list (empty)
- `GET /api/budget` → budget remaining from env

## SQLModel + SQLite

**Docs:** [Create a table / engine](https://sqlmodel.tiangolo.com/tutorial/create-db-and-table/), [Session with dependency](https://sqlmodel.tiangolo.com/tutorial/fastapi/session-with-dependency/), [Read data](https://sqlmodel.tiangolo.com/tutorial/fastapi/read/)

```python
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlmodel import Field, Session, SQLModel, create_engine, select

class Experiment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    status: str = "pending"

sqlite_url = "sqlite:///./robolab.db"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session

SessionDep = Annotated[Session, Depends(get_session)]

@app.on_event("startup")  # or lifespan context — see FastAPI lifespan docs
def on_startup() -> None:
    create_db_and_tables()

@app.get("/api/experiments")
def list_experiments(session: SessionDep) -> dict:
    rows = list(session.exec(select(Experiment)).all())
    return {
        "experiments": [{"id": r.id, "name": r.name} for r in rows],
        "count": len(rows),
    }
```

Phase 0 response is always `{"experiments": [], "count": 0}` (no seed rows).


Notes from docs:

- `table=True` marks a table model.
- `Field(default=None, primary_key=True)` for autoincrement id.
- SQLite + FastAPI needs `connect_args={"check_same_thread": False}`.
- Prefer FastAPI lifespan over deprecated `@app.on_event` if using current FastAPI; both work for Phase 0.

## Budget from env

**Docs:** [uv `--env-file`](https://docs.astral.sh/uv/reference/cli/), Python [`os.getenv`](https://docs.python.org/3/library/os.html#os.getenv)

No extra dotenv package. `make dev` runs the API via `uv run --env-file .env` (falls back if `.env` missing; use `.env.example` values by copying).

```python
import os

def get_budget() -> dict[str, float]:
    cap = float(os.environ.get("BUDGET_USD_CAP", "100"))
    month_spend = 0.0  # Phase 0: no cost ledger yet
    return {
        "budget_usd_cap": cap,
        "month_spend_usd": month_spend,
        "remaining_usd": cap - month_spend,
    }
```

## uvicorn

**Docs:** [Uvicorn deployment / CLI](https://www.uvicorn.org/)

```bash
uv run --package robolab-api uvicorn robolab_api.main:app --reload --host 127.0.0.1 --port 8000
```

Signature: `uvicorn <module>:<app> --reload --host HOST --port PORT`

## React + Vite + TypeScript

**Docs:** [Vite Getting Started](https://vite.dev/guide/), [create-vite templates](https://vite.dev/guide/#scaffolding-your-first-vite-project)

```bash
npm create vite@latest web -- --template react-ts
```

Default dev server: `http://localhost:5173` (`npm run dev` → `vite`).

Proxy API (optional; CORS also configured on backend):

```ts
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
```

Fetch:

```ts
const res = await fetch("/api/experiments");
const experiments: Experiment[] = await res.json();
```

## Tailwind CSS v4 (Vite plugin)

**Docs:** [Installing Tailwind CSS with Vite](https://tailwindcss.com/docs/installation/using-vite)

```bash
npm install tailwindcss @tailwindcss/vite
```

```ts
import tailwindcss from "@tailwindcss/vite";
// plugins: [react(), tailwindcss()]
```

```css
@import "tailwindcss";
```

## Recharts

**Docs:** [Recharts](https://recharts.org/en-US/)

Not required for Phase 0 empty table. Skip until Phase 1+ charts.

## make / concurrent processes

No extra process manager dep. Makefile:

```makefile
dev:
	@trap 'kill 0' EXIT; \
	uv run --env-file .env --package robolab-api \
	  uvicorn robolab_api.main:app --reload --host 127.0.0.1 --port 8000 & \
	cd web && npm run dev
```

## Ports (Phase 0)

| Service | Port |
|---|---|
| Vite dashboard | `5173` |
| FastAPI | `8000` |

## Sources

- [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [uv projects guide](https://docs.astral.sh/uv/guides/projects/)
- [uv dependencies / workspace sources](https://docs.astral.sh/uv/concepts/projects/dependencies/)
- [FastAPI first steps](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/)
- [FastAPI dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [SQLModel create table](https://sqlmodel.tiangolo.com/tutorial/create-db-and-table/)
- [SQLModel FastAPI session dependency](https://sqlmodel.tiangolo.com/tutorial/fastapi/session-with-dependency/)
- [Vite guide](https://vite.dev/guide/)
- [Tailwind + Vite](https://tailwindcss.com/docs/installation/using-vite)
