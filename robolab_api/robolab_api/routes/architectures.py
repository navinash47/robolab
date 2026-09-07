"""Saved architecture Builder CRUD + builtin metadata."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlmodel import select

from robolab.compute.local import repo_root
from robolab.core.arch import list_archs
from robolab_api.db import SavedArch, SessionDep

router = APIRouter(tags=["architectures"])

_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

BUILTIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "mlp": {"hidden_sizes": [64, 64], "activation": "tanh"},
    "kan": {"hidden_sizes": [32, 32], "grid_size": 5, "spline_order": 3},
    "kaf": {
        "hidden_sizes": [64, 64],
        "num_grids": 8,
        "activation_expectation": 1.64,
        "use_layernorm": True,
        "spline_dropout": 0.0,
        "lr_default": 3.0e-4,
    },
    "gpkan": {
        "hidden_sizes": [32, 32],
        "num_basis": 8,
        "init_bandwidth": 1.0,
        "base_activation": "gelu",
        "lr_default": 3.0e-4,
    },
    "fan": {
        "hidden_sizes": [64, 64],
        "p_ratio": 0.25,
        "activation": "gelu",
        "lr_default": 3.0e-4,
    },
    "avinash_wall": {
        "algorithm": "q_learning",
        "alpha": 0.1,
        "gamma": 1.0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.05,
        "explore_episodes": 200,
        "episode_max_steps": 1200,
        "linear_vel": 0.3,
        "angular_vel": 0.7,
        "near_max": 0.7,
        "medium_max": 0.9,
        "lr_default": 0.1,
    },
}

BUILTIN_META: dict[str, dict[str, str]] = {
    "mlp": {
        "label": "MLP",
        "paper": "baseline",
        "blurb": "Classic multilayer perceptron — dense layers with a fixed activation.",
        "docs_url": "https://github.com/navinash47/robolab/blob/main/docs/PHASE_ARCH_BUILDER_APIS.md",
    },
    "kan": {
        "label": "KAN (B-spline)",
        "paper": "Liu et al. / efficient-kan",
        "blurb": "Kolmogorov–Arnold network with learnable B-spline edge functions.",
        "docs_url": "https://github.com/navinash47/robolab/blob/main/docs/PHASE_ARCH_BUILDER_APIS.md",
    },
    "kaf": {
        "label": "KAF (Kolmogorov-Arnold Fourier)",
        "paper": "arXiv:2502.06018",
        "blurb": "RFF + GELU hybrid from Kolmogorov–Arnold Fourier Networks (arXiv:2502.06018).",
        "docs_url": "https://arxiv.org/abs/2502.06018",
    },
    "gpkan": {
        "label": "GPKAN (Gaussian RBF KAN)",
        "paper": "KAF baseline / GP-KAN spirit arXiv:2407.18397",
        "blurb": "Gaussian RBF–KAN baseline — GELU base plus learnable RBF centers per edge.",
        "docs_url": "https://arxiv.org/abs/2407.18397",
    },
    "fan": {
        "label": "FAN (Fourier Analysis Network)",
        "paper": "Dong et al. arXiv:2410.02675",
        "blurb": "Fourier Analysis Network–style layer: cos/sin path plus nonlinear σ path.",
        "docs_url": "https://arxiv.org/abs/2410.02675",
    },
    "avinash_wall": {
        "label": "Avinash Wall Follow",
        "paper": "Course P2_D3 — Q-learning wall follow",
        "blurb": "Tabular Q-learning wall follower (27 states × 3 actions) from the Robotics course project — PDF reward, ε-greedy, α=0.1, γ=1.0.",
        "docs_url": "https://github.com/navinash47/robolab/blob/main/docs/AVINASH_WALL.md",
    },
}


class ArchCreate(BaseModel):
    name: str
    base_arch: str
    cfg: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        v = v.strip()
        if not _NAME_RE.match(v):
            raise ValueError("name must match [a-zA-Z0-9_-]{1,64}")
        return v


class ArchUpdate(BaseModel):
    name: str | None = None
    cfg: dict[str, Any] | None = None
    notes: str | None = None

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not _NAME_RE.match(v):
            raise ValueError("name must match [a-zA-Z0-9_-]{1,64}")
        return v


def default_cfg(base_arch: str) -> dict[str, Any]:
    return dict(BUILTIN_DEFAULTS.get(base_arch, {"hidden_sizes": [64, 64]}))


def _row_to_dict(row: SavedArch) -> dict[str, Any]:
    try:
        cfg = json.loads(row.cfg_json)
    except json.JSONDecodeError:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "id": row.id,
        "name": row.name,
        "base_arch": row.base_arch,
        "cfg": cfg,
        "notes": row.notes or "",
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _ensure_builtin(base_arch: str) -> None:
    import robolab.archs  # noqa: F401

    registered = list_archs()
    if base_arch not in registered:
        raise HTTPException(
            400,
            f"Unknown base_arch={base_arch!r}. Registered: {registered}",
        )


def _write_yaml_mirror(name: str, base_arch: str, cfg: dict[str, Any], notes: str) -> None:
    root = repo_root() / "configs" / "arches"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.yaml"
    payload = {
        "name": name,
        "base_arch": base_arch,
        "arch_cfg": cfg,
        "notes": notes or "",
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _delete_yaml_mirror(name: str) -> None:
    path = repo_root() / "configs" / "arches" / f"{name}.yaml"
    if path.is_file():
        path.unlink()


def resolve_arch_for_run(
    session: SessionDep,
    arch: str,
    arch_cfg: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], str | None]:
    """Return (base_arch, arch_cfg, saved_name).

    Accepts builtin names, ``custom:<id>``, or a unique saved ``name``.
    """
    import robolab.archs  # noqa: F401

    registered = set(list_archs())
    raw = (arch or "").strip()
    cfg = dict(arch_cfg or {})

    if raw.startswith("custom:"):
        sid = raw.split(":", 1)[1].strip()
        row = session.get(SavedArch, sid)
        if not row:
            raise HTTPException(400, f"Unknown saved architecture id={sid!r}")
        try:
            saved_cfg = json.loads(row.cfg_json)
        except json.JSONDecodeError:
            saved_cfg = {}
        merged = {**default_cfg(row.base_arch), **(saved_cfg if isinstance(saved_cfg, dict) else {}), **cfg}
        return row.base_arch, merged, row.name

    if raw in registered:
        merged = {**default_cfg(raw), **cfg} if cfg else default_cfg(raw)
        # If client sent a non-empty cfg, prefer it over defaults (still fill missing keys)
        if cfg:
            merged = {**default_cfg(raw), **cfg}
        return raw, merged, None

    # Unique saved name
    row = session.exec(select(SavedArch).where(SavedArch.name == raw)).first()
    if row:
        try:
            saved_cfg = json.loads(row.cfg_json)
        except json.JSONDecodeError:
            saved_cfg = {}
        merged = {**default_cfg(row.base_arch), **(saved_cfg if isinstance(saved_cfg, dict) else {}), **cfg}
        return row.base_arch, merged, row.name

    raise HTTPException(
        400,
        f"Unknown arch={raw!r}. Use a builtin {sorted(registered)}, "
        "custom:<id>, or a saved architecture name.",
    )


@router.get("/api/architectures")
def list_architectures(session: SessionDep) -> dict:
    import robolab.archs  # noqa: F401

    builtins = []
    for name in list_archs():
        meta = BUILTIN_META.get(
            name, {"label": name, "paper": "", "blurb": "", "docs_url": ""}
        )
        builtins.append(
            {
                "name": name,
                "label": meta.get("label", name),
                "paper": meta.get("paper", ""),
                "blurb": meta.get("blurb", ""),
                "docs_url": meta.get("docs_url", ""),
                "cfg": default_cfg(name),
                "builtin": True,
            }
        )
    rows = session.exec(select(SavedArch).order_by(SavedArch.updated_at.desc())).all()
    return {"builtins": builtins, "saved": [_row_to_dict(r) for r in rows]}


@router.post("/api/architectures")
def create_architecture(body: ArchCreate, session: SessionDep) -> dict:
    _ensure_builtin(body.base_arch)
    existing = session.exec(select(SavedArch).where(SavedArch.name == body.name)).first()
    if existing:
        raise HTTPException(400, f"Architecture name {body.name!r} already exists")
    cfg = {**default_cfg(body.base_arch), **(body.cfg or {})}
    # Drop optional UI-only hint from affecting nothing — keep in cfg for round-trip
    arch_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    row = SavedArch(
        id=arch_id,
        name=body.name,
        base_arch=body.base_arch,
        cfg_json=json.dumps(cfg),
        notes=body.notes or "",
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    try:
        _write_yaml_mirror(row.name, row.base_arch, cfg, row.notes)
    except OSError:
        pass
    return _row_to_dict(row)


@router.get("/api/architectures/{arch_id}")
def get_architecture(arch_id: str, session: SessionDep) -> dict:
    row = session.get(SavedArch, arch_id)
    if not row:
        raise HTTPException(404, f"Architecture {arch_id} not found")
    return _row_to_dict(row)


@router.put("/api/architectures/{arch_id}")
def update_architecture(arch_id: str, body: ArchUpdate, session: SessionDep) -> dict:
    row = session.get(SavedArch, arch_id)
    if not row:
        raise HTTPException(404, f"Architecture {arch_id} not found")
    old_name = row.name
    if body.name is not None and body.name != row.name:
        clash = session.exec(select(SavedArch).where(SavedArch.name == body.name)).first()
        if clash:
            raise HTTPException(400, f"Architecture name {body.name!r} already exists")
        row.name = body.name
    if body.cfg is not None:
        cfg = {**default_cfg(row.base_arch), **body.cfg}
        row.cfg_json = json.dumps(cfg)
    if body.notes is not None:
        row.notes = body.notes
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()
    session.refresh(row)
    try:
        if old_name != row.name:
            _delete_yaml_mirror(old_name)
        cfg = json.loads(row.cfg_json)
        _write_yaml_mirror(row.name, row.base_arch, cfg if isinstance(cfg, dict) else {}, row.notes)
    except (OSError, json.JSONDecodeError):
        pass
    return _row_to_dict(row)


@router.delete("/api/architectures/{arch_id}")
def delete_architecture(arch_id: str, session: SessionDep) -> dict:
    row = session.get(SavedArch, arch_id)
    if not row:
        raise HTTPException(404, f"Architecture {arch_id} not found")
    name = row.name
    session.delete(row)
    session.commit()
    try:
        _delete_yaml_mirror(name)
    except OSError:
        pass
    return {"ok": True, "id": arch_id, "name": name}
