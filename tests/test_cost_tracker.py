"""Unit tests for CostTracker with budget enforcement."""
from __future__ import annotations

import json

import pytest

from swe_af.execution.cost_tracker import BudgetExceeded, CostTracker, _current_cost_tracker, _track_cost


def test_record_accumulates():
    t = CostTracker(max_cost_usd=10.0)
    t.record_sync("coder", 0.5, 1000)
    t.record_sync("qa", 0.3, 500)
    assert t.total_cost_usd == pytest.approx(0.8)
    assert t.total_tokens == 1500
    assert t.total_invocations == 2


def test_budget_exceeded():
    t = CostTracker(max_cost_usd=1.0)
    t.record_sync("coder", 0.5)
    with pytest.raises(BudgetExceeded):
        t.record_sync("coder", 0.6)


def test_summary():
    t = CostTracker(max_cost_usd=5.0)
    t.record_sync("coder", 1.0, 5000)
    t.record_sync("qa", 0.5, 2000)
    s = t.summary()
    assert s["total_cost_usd"] == 1.5
    assert s["budget_remaining"] == 3.5
    assert s["budget_percent"] == 30.0
    assert s["total_tokens"] == 7000
    assert s["total_invocations"] == 2
    assert "coder" in s["cost_by_agent"]
    # cost_by_agent should be sorted descending
    agents = list(s["cost_by_agent"].keys())
    assert agents[0] == "coder"


def test_contextvar():
    t = CostTracker(max_cost_usd=10.0)
    token = _current_cost_tracker.set(t)
    try:
        assert _current_cost_tracker.get() is t
    finally:
        _current_cost_tracker.reset(token)


def test_flush_writes_file(tmp_path):
    t = CostTracker(max_cost_usd=10.0, artifacts_dir=str(tmp_path))
    t.record_sync("coder", 0.5)
    data = json.loads((tmp_path / "cost_status.json").read_text())
    assert data["total_cost_usd"] == 0.5
    assert data["max_cost_usd"] == 10.0


def test_flush_creates_directory(tmp_path):
    nested = tmp_path / "deep" / "nested"
    t = CostTracker(max_cost_usd=10.0, artifacts_dir=str(nested))
    t.record_sync("coder", 0.1)
    assert (nested / "cost_status.json").exists()


def test_no_flush_without_artifacts_dir():
    """CostTracker without artifacts_dir should not error on flush."""
    t = CostTracker(max_cost_usd=10.0)
    t.record_sync("coder", 0.5)  # Should not raise


def test_budget_exactly_at_limit():
    """Budget check triggers at >= max_cost_usd."""
    t = CostTracker(max_cost_usd=1.0)
    with pytest.raises(BudgetExceeded):
        t.record_sync("coder", 1.0)


def test_cost_by_agent_aggregation():
    t = CostTracker(max_cost_usd=100.0)
    t.record_sync("coder", 1.0, 100)
    t.record_sync("coder", 2.0, 200)
    t.record_sync("qa", 0.5, 50)
    assert t.cost_by_agent["coder"] == pytest.approx(3.0)
    assert t.tokens_by_agent["coder"] == 300
    assert t.cost_by_agent["qa"] == pytest.approx(0.5)


class _FakeMetrics:
    def __init__(self, cost, input_tokens, output_tokens):
        self.total_cost_usd = cost
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, cost=0.0, input_tokens=0, output_tokens=0):
        self.metrics = _FakeMetrics(cost, input_tokens, output_tokens)


def test_track_cost_records_to_active_tracker():
    t = CostTracker(max_cost_usd=100.0)
    token = _current_cost_tracker.set(t)
    try:
        resp = _FakeResponse(cost=0.25, input_tokens=100, output_tokens=50)
        _track_cost("coder", resp)
        assert t.total_cost_usd == pytest.approx(0.25)
        assert t.total_tokens == 150
    finally:
        _current_cost_tracker.reset(token)


def test_track_cost_noop_without_tracker():
    """_track_cost should silently no-op when no tracker is set."""
    resp = _FakeResponse(cost=0.25)
    _track_cost("coder", resp)  # Should not raise


def test_track_cost_ignores_zero_cost():
    t = CostTracker(max_cost_usd=100.0)
    token = _current_cost_tracker.set(t)
    try:
        resp = _FakeResponse(cost=0.0)
        _track_cost("coder", resp)
        assert t.total_invocations == 0
    finally:
        _current_cost_tracker.reset(token)


def test_track_cost_handles_none_response():
    t = CostTracker(max_cost_usd=100.0)
    token = _current_cost_tracker.set(t)
    try:
        _track_cost("coder", None)
        assert t.total_invocations == 0
    finally:
        _current_cost_tracker.reset(token)
