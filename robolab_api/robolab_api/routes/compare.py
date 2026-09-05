"""Compare runs: fetch real W&B history and param counts."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from robolab.core.arch import get_arch
from robolab_api.db import Run, SessionDep

router = APIRouter(prefix="/api/compare", tags=["compare"])

METRIC_KEY = "rollout/ep_rew_mean"


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=1)


class HistoryPoint(BaseModel):
    step: int
    mean_return: float | None


class CompareRun(BaseModel):
    id: str
    name: str
    arch: str
    status: str
    param_count: int | None = None
    mean_return: float | None = None
    wandb_url: str | None = None
    wandb_run_id: str | None = None
    history: list[HistoryPoint] = Field(default_factory=list)
    error: str | None = None


class CompareResponse(BaseModel):
    metric_key: str = METRIC_KEY
    runs: list[CompareRun]


def parse_wandb_path(wandb_url: str) -> tuple[str, str, str]:
    """Parse https://wandb.ai/{entity}/{project}/runs/{run_id} → entity, project, run_id."""
    path = urlparse(wandb_url).path.strip("/")
    parts = path.split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise ValueError(f"Unrecognized wandb_url path: {wandb_url}")
    return parts[0], parts[1], parts[3]


def _obs_dim_fallback(cfg: dict[str, Any]) -> int:
    """wall_follow lidar default; override via config if present."""
    domain = cfg.get("domain") or {}
    if "obs_dim" in cfg:
        return int(cfg["obs_dim"])
    # diffdrive_lidar wall_follow: typical 5 beams + maybe extras — read task if set
    return int(cfg.get("arch_cfg", {}).get("obs_dim", 5))


def _act_dim_fallback(cfg: dict[str, Any]) -> int:
    return int(cfg.get("arch_cfg", {}).get("act_dim", 2))


def compute_param_count(row: Run) -> int | None:
    if row.param_count is not None:
        return int(row.param_count)
    try:
        cfg = json.loads(row.config_json) if row.config_json else {}
        arch_cfg = dict(cfg.get("arch_cfg") or {})
        obs_dim = _obs_dim_fallback(cfg)
        act_dim = _act_dim_fallback(cfg)
        arch_cls = get_arch(row.arch)
        net = arch_cls(obs_dim=obs_dim, act_dim=act_dim, cfg=arch_cfg)
        return int(net.param_count())
    except Exception:
        return None


def fetch_wandb_history(wandb_url: str) -> tuple[list[HistoryPoint], int | None]:
    """Return history points + optional param_count from W&B config/summary."""
    import wandb

    entity, project, rid = parse_wandb_path(wandb_url)
    api = wandb.Api(timeout=60)
    wrun = api.run(f"{entity}/{project}/{rid}")

    param_from_wb: int | None = None
    for source in (wrun.summary, wrun.config):
        try:
            raw = source.get("param_count") if hasattr(source, "get") else None
            if raw is not None:
                param_from_wb = int(raw)
                break
        except (TypeError, ValueError):
            continue

    # pandas=False → list[dict]; avoids optional pandas / empty-DF traps
    raw_hist = wrun.history(
        samples=500,
        keys=[METRIC_KEY],
        x_axis="_step",
        pandas=False,
    )
    points: list[HistoryPoint] = []
    if isinstance(raw_hist, list):
        rows: list[Any] = raw_hist
    elif raw_hist is None:
        rows = []
    else:
        try:
            rows = raw_hist.to_dict(orient="records")
        except Exception:
            rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if METRIC_KEY not in row or row[METRIC_KEY] is None:
            continue
        try:
            y = float(row[METRIC_KEY])
            step = int(row["_step"])
        except (TypeError, ValueError, KeyError):
            continue
        points.append(HistoryPoint(step=step, mean_return=y))
    points.sort(key=lambda p: p.step)
    return points, param_from_wb


@router.post("", response_model=CompareResponse)
def compare_runs(body: CompareRequest, session: SessionDep) -> CompareResponse:
    if not os.environ.get("WANDB_API_KEY", "").strip():
        raise HTTPException(
            400,
            "WANDB_API_KEY missing — cannot fetch real W&B history for Compare.",
        )

    # Preserve request order; skip unknown ids with error entries
    results: list[CompareRun] = []
    for run_id in body.run_ids:
        row = session.get(Run, run_id)
        if not row:
            results.append(
                CompareRun(
                    id=run_id,
                    name="?",
                    arch="?",
                    status="MISSING",
                    error=f"Run {run_id} not found",
                )
            )
            continue

        local_params = compute_param_count(row)
        wandb_run_id: str | None = None
        history: list[HistoryPoint] = []
        err: str | None = None
        wb_params: int | None = None

        if not row.wandb_url:
            err = "No wandb_url on this run — cannot fetch history"
        else:
            try:
                wandb_run_id = parse_wandb_path(row.wandb_url)[2]
                history, wb_params = fetch_wandb_history(row.wandb_url)
                if not history:
                    err = f"W&B history empty for metric {METRIC_KEY}"
            except Exception as exc:
                err = f"W&B history fetch failed: {type(exc).__name__}: {exc}"

        param_count = local_params if local_params is not None else wb_params
        # Persist computed count for next time
        if param_count is not None and row.param_count is None:
            row.param_count = param_count
            session.add(row)
            session.commit()

        results.append(
            CompareRun(
                id=row.id,
                name=row.name,
                arch=row.arch,
                status=row.status,
                param_count=param_count,
                mean_return=row.mean_return,
                wandb_url=row.wandb_url,
                wandb_run_id=wandb_run_id,
                history=history,
                error=err,
            )
        )

    return CompareResponse(metric_key=METRIC_KEY, runs=results)


@router.get("/archs")
def list_registered_archs() -> dict:
    """Convenience; primary list is GET /api/archs on main app."""
    from robolab.core.arch import list_archs

    return {"archs": list_archs()}
