"""Unit tests for Phase 6 transfer metrics helpers."""

from __future__ import annotations

from robolab.train.transfer import _mean_std_ci, gap, transfer_ratio


def test_transfer_ratio_and_gap():
    assert transfer_ratio(10.0, 8.0) == 0.8
    assert gap(10.0, 8.0) == 2.0
    assert transfer_ratio(0.0, 1.0) is None
    assert transfer_ratio(None, 1.0) is None
    assert gap(None, 1.0) is None


def test_mean_std_ci_basic():
    stats = _mean_std_ci([1.0, 3.0, 5.0])
    assert stats["n"] == 3
    assert abs(stats["mean"] - 3.0) < 1e-9
    assert stats["ci_low"] is not None
    assert stats["ci_high"] is not None
    empty = _mean_std_ci([])
    assert empty["mean"] is None
    one = _mean_std_ci([4.0])
    assert one["mean"] == 4.0
    assert one["ci_low"] is None
