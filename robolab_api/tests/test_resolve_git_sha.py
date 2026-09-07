"""Unit tests for RunPod git SHA resolution (dirty tree must not refuse)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robolab.compute.runpod import RunPodConfigError, resolve_git_sha


def _proc(stdout: str = "", returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


def test_dirty_tree_falls_back_to_origin_main(tmp_path: Path) -> None:
    """Dirty working tree must still resolve a SHA (origin/main) — never raise."""
    head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    main = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def fake_git(args: list[str], root: Path):
        cmd = args[0] if args else ""
        if cmd == "remote":
            return _proc("origin\n")
        if cmd == "status":
            return _proc("?? configs/arches/kaf_wall_follow.yaml\n")
        if cmd == "rev-parse" and args[1:] == ["HEAD"]:
            return _proc(f"{head}\n")
        if cmd == "rev-parse" and args[1:] == ["origin/main"]:
            return _proc(f"{main}\n")
        if cmd == "rev-parse" and args[1:] == ["--abbrev-ref", "@{u}"]:
            return _proc(returncode=128, stderr="no upstream")
        if cmd == "ls-remote":
            # HEAD not on remote; origin/main tip is.
            return _proc(f"{main}\trefs/heads/main\n")
        return _proc(returncode=128, stderr=f"unexpected: {args}")

    with patch("robolab.compute.runpod._git", side_effect=fake_git):
        with patch("robolab.compute.runpod._note_dirty_git_fallback"):
            with patch.dict("os.environ", {"GIT_SHA": ""}, clear=False):
                assert resolve_git_sha(tmp_path) == main


def test_dirty_tree_uses_pushed_head(tmp_path: Path) -> None:
    """Dirty tree with HEAD already on remote → still launch with HEAD (warn only)."""
    head = "cccccccccccccccccccccccccccccccccccccccc"

    def fake_git(args: list[str], root: Path):
        cmd = args[0] if args else ""
        if cmd == "remote":
            return _proc("origin\n")
        if cmd == "status":
            return _proc(" M robolab/robolab/compute/runpod.py\n")
        if cmd == "rev-parse" and args[1:] == ["HEAD"]:
            return _proc(f"{head}\n")
        if cmd == "ls-remote":
            return _proc(f"{head}\trefs/heads/main\n")
        return _proc(returncode=128, stderr=f"unexpected: {args}")

    with patch("robolab.compute.runpod._git", side_effect=fake_git):
        with patch("robolab.compute.runpod._note_dirty_git_fallback") as note:
            with patch.dict("os.environ", {"GIT_SHA": ""}, clear=False):
                assert resolve_git_sha(tmp_path) == head
            note.assert_called_once()


def test_clean_head_on_remote_pins_head(tmp_path: Path) -> None:
    head = "dddddddddddddddddddddddddddddddddddddddd"

    def fake_git(args: list[str], root: Path):
        cmd = args[0] if args else ""
        if cmd == "remote":
            return _proc("origin\n")
        if cmd == "status":
            return _proc("")
        if cmd == "rev-parse" and args[1:] == ["HEAD"]:
            return _proc(f"{head}\n")
        if cmd == "ls-remote":
            return _proc(f"{head}\trefs/heads/main\n")
        return _proc(returncode=128, stderr=f"unexpected: {args}")

    with patch("robolab.compute.runpod._git", side_effect=fake_git):
        with patch.dict("os.environ", {"GIT_SHA": ""}, clear=False):
            assert resolve_git_sha(tmp_path) == head


def test_git_sha_env_override(tmp_path: Path) -> None:
    pinned = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

    def fake_git(args: list[str], root: Path):
        cmd = args[0] if args else ""
        if cmd == "remote":
            return _proc("origin\n")
        if cmd == "ls-remote":
            return _proc(f"{pinned}\trefs/heads/main\n")
        return _proc(returncode=128, stderr=f"unexpected: {args}")

    with patch("robolab.compute.runpod._git", side_effect=fake_git):
        with patch.dict("os.environ", {"GIT_SHA": pinned}, clear=False):
            assert resolve_git_sha(tmp_path) == pinned


def test_no_remote_still_refuses(tmp_path: Path) -> None:
    def fake_git(args: list[str], root: Path):
        if args and args[0] == "remote":
            return _proc("")
        return _proc(returncode=128, stderr="unexpected")

    with patch("robolab.compute.runpod._git", side_effect=fake_git):
        with patch.dict("os.environ", {"GIT_SHA": ""}, clear=False):
            with pytest.raises(RunPodConfigError, match="No git remote"):
                resolve_git_sha(tmp_path)
