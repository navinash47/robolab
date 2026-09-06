"""Unit tests for BACKEND_PUBLIC_URL preflight (no live tunnel required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robolab.compute.runpod import (
    RunPodConfigError,
    _latest_tunnel_url_from_logs,
    preflight_backend_public_url,
)


def test_preflight_rejects_localhost() -> None:
    with pytest.raises(RunPodConfigError, match="not localhost"):
        preflight_backend_public_url("http://127.0.0.1:8000")


def test_preflight_rejects_missing() -> None:
    with patch.dict("os.environ", {"BACKEND_PUBLIC_URL": ""}, clear=False):
        with pytest.raises(RunPodConfigError, match="missing"):
            preflight_backend_public_url("")


def test_preflight_ok_on_200(tmp_path: Path) -> None:
    resp = MagicMock()
    resp.status = 200
    resp.getcode.return_value = 200
    resp.read.return_value = b'{"ok":true}'
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        patch("robolab.compute.runpod._CF_LOG_CANDIDATES", (tmp_path / "missing.log",)),
        patch("urllib.request.urlopen", return_value=resp) as urlopen,
    ):
        out = preflight_backend_public_url("https://example.trycloudflare.com")
    assert out == "https://example.trycloudflare.com"
    assert urlopen.called
    req = urlopen.call_args[0][0]
    assert req.full_url.endswith("/health")


def test_preflight_rejects_non_200() -> None:
    resp = MagicMock()
    resp.status = 502
    resp.getcode.return_value = 502
    resp.read.return_value = b"bad gateway"
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        patch("robolab.compute.runpod._CF_LOG_CANDIDATES", ()),
        patch("urllib.request.urlopen", return_value=resp),
        pytest.raises(RunPodConfigError, match="HTTP 502"),
    ):
        preflight_backend_public_url("https://stale.trycloudflare.com")


def test_preflight_mismatch_log_ok_when_health_200(tmp_path: Path) -> None:
    """Leftover cf log must not block a healthy configured URL."""
    log = tmp_path / "robolab-cloudflared.log"
    log.write_text("https://fresh-abc.trycloudflare.com\n")
    resp = MagicMock()
    resp.status = 200
    resp.getcode.return_value = 200
    resp.read.return_value = b'{"ok":true}'
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        patch("robolab.compute.runpod._CF_LOG_CANDIDATES", (log,)),
        patch("urllib.request.urlopen", return_value=resp),
    ):
        out = preflight_backend_public_url("https://old-xyz.trycloudflare.com")
    assert out == "https://old-xyz.trycloudflare.com"


def test_preflight_includes_stale_hint_on_health_fail(tmp_path: Path) -> None:
    log = tmp_path / "robolab-cloudflared.log"
    log.write_text("https://fresh-abc.trycloudflare.com\n")
    resp = MagicMock()
    resp.status = 530
    resp.getcode.return_value = 530
    resp.read.return_value = b"down"
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False

    with (
        patch("robolab.compute.runpod._CF_LOG_CANDIDATES", (log,)),
        patch("urllib.request.urlopen", return_value=resp),
        pytest.raises(RunPodConfigError, match="fresh-abc"),
    ):
        preflight_backend_public_url("https://old-xyz.trycloudflare.com")


def test_latest_tunnel_url_from_logs(tmp_path: Path) -> None:
    log = tmp_path / "cf.log"
    log.write_text("first https://one.trycloudflare.com then https://two.trycloudflare.com\n")
    with patch("robolab.compute.runpod._CF_LOG_CANDIDATES", (log,)):
        assert _latest_tunnel_url_from_logs() == "https://two.trycloudflare.com"
