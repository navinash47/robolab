#!/usr/bin/env python3
"""Durable RoboLab API on :8000 (no --reload). Used by LaunchAgent com.robolab.api8000."""

from __future__ import annotations

import os
import sys
from pathlib import Path

root = Path("/Users/avinashnandyala/Projects/robolab")
os.chdir(root)


def load_env() -> None:
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k] = v


load_env()
sys.path.insert(0, str(root / "robolab_api"))
sys.path.insert(0, str(root / "robolab"))

from robolab_api.main import app  # noqa: E402


@app.middleware("http")
async def refresh_env_middleware(request, call_next):
    # Pick up rotated BACKEND_PUBLIC_URL without full process restart.
    try:
        load_env()
    except Exception:
        pass
    return await call_next(request)


import uvicorn

uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
