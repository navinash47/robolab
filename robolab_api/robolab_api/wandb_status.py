"""W&B credential presence / shape / auth probe (never logs the key)."""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.request
from typing import Any

# Legacy keys are 40 hex; current cloud keys are wandb_v1_ + 77 charset chars (86 total).
_LEGACY_KEY = re.compile(r"^[a-f0-9]{40}$")
_V1_KEY = re.compile(r"^wandb_v1_[A-Za-z0-9_]{77}$")


def key_well_formed(key: str) -> bool:
    return bool(_LEGACY_KEY.fullmatch(key) or _V1_KEY.fullmatch(key))


def key_shape_hint(key: str) -> str | None:
    if not key:
        return "WANDB_API_KEY is empty"
    if key.startswith("wandb_v1_"):
        body = len(key) - len("wandb_v1_")
        if body != 77:
            return (
                f"WANDB_API_KEY looks truncated/corrupt: wandb_v1_ body len={body} "
                f"(expected 77; total expected 86, got {len(key)}). "
                "Paste a fresh key from https://wandb.ai/authorize"
            )
        if not _V1_KEY.fullmatch(key):
            return "WANDB_API_KEY has unexpected characters for wandb_v1_ format"
        return None
    if len(key) != 40:
        return (
            f"WANDB_API_KEY length {len(key)} is neither legacy 40-char nor "
            "wandb_v1_ (86-char). Paste a fresh key from https://wandb.ai/authorize"
        )
    if not _LEGACY_KEY.fullmatch(key):
        return "WANDB_API_KEY is 40 chars but not hex (legacy format)"
    return None


def probe_viewer(key: str, timeout_s: float = 8.0) -> tuple[bool | None, str]:
    """Auth-check key via W&B GraphQL viewer query. Returns (ok, detail).

    Legacy REST GET /viewer now 404s on api.wandb.ai; GraphQL is the live probe.
    """
    body = b'{"query":"{ viewer { id } }"}'
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "robolab/wandb-status",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()[:500].decode("utf-8", "replace").replace(key, "[REDACTED]")
            # Valid keys return data.viewer; invalid often still HTTP 200 with errors
            if '"viewer"' in raw and '"errors"' not in raw:
                return True, f"graphql viewer HTTP {resp.status}"
            if '"errors"' in raw or '"viewer":null' in raw.replace(" ", ""):
                return False, f"graphql viewer rejected: {raw[:180]}"
            return True, f"graphql viewer HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        err_body = exc.read()[:200].decode("utf-8", "replace")
        err_body = err_body.replace(key, "[REDACTED]")
        return False, f"graphql HTTP {exc.code}: {err_body}"
    except Exception as exc:  # noqa: BLE001 — surface probe failures cleanly
        return None, f"{type(exc).__name__}: {exc}"


def get_wandb_status(*, probe: bool = True) -> dict[str, Any]:
    raw = os.environ.get("WANDB_API_KEY", "")
    key = raw.strip()
    present = bool(key)
    well_formed = key_well_formed(key) if present else False
    hint = key_shape_hint(key) if present else "WANDB_API_KEY not set in process env"
    project = (os.environ.get("WANDB_PROJECT") or "robolab").strip() or "robolab"
    entity = (os.environ.get("WANDB_ENTITY") or "").strip() or None
    mode = (os.environ.get("WANDB_MODE") or "").strip() or None

    auth_ok: bool | None = None
    detail = hint or "ok"
    if present and probe:
        auth_ok, detail = probe_viewer(key)
        if auth_ok is False and hint:
            detail = f"{hint}; {detail}"

    status = "missing"
    if present and auth_ok is True:
        status = "ok"
    elif present and not well_formed:
        status = "invalid_shape"
    elif present and auth_ok is False:
        status = "invalid"
    elif present and auth_ok is None:
        status = "unreachable"

    return {
        "present": present,
        "well_formed": well_formed,
        "key_len": len(key) if present else 0,
        "auth_ok": auth_ok,
        "status": status,
        "project": project,
        "entity": entity,
        "mode": mode,
        "detail": detail,
    }
