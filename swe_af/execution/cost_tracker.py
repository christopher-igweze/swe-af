"""Build-level cost tracking with budget enforcement."""
from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from pathlib import Path


class BudgetExceeded(Exception):
    pass


class CostTracker:
    def __init__(self, max_cost_usd: float = 50.0, artifacts_dir: str = "") -> None:
        self.max_cost_usd = max_cost_usd
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._lock = asyncio.Lock()
        self._start_time = time.time()

        self.total_cost_usd: float = 0.0
        self.total_tokens: int = 0
        self.total_invocations: int = 0
        self.cost_by_agent: dict[str, float] = {}
        self.tokens_by_agent: dict[str, int] = {}

    async def record(self, agent_name: str, cost_usd: float, tokens: int = 0) -> None:
        """Record cost from an agent call (async, thread-safe)."""
        async with self._lock:
            self._accumulate(agent_name, cost_usd, tokens)
            self._flush()

        if self.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Budget exceeded: ${self.total_cost_usd:.2f} >= ${self.max_cost_usd:.2f}"
            )

    def record_sync(self, agent_name: str, cost_usd: float, tokens: int = 0) -> None:
        """Record cost from an agent call (sync, for non-async contexts)."""
        self._accumulate(agent_name, cost_usd, tokens)
        self._flush()

        if self.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Budget exceeded: ${self.total_cost_usd:.2f} >= ${self.max_cost_usd:.2f}"
            )

    def summary(self) -> dict:
        """Return a snapshot of cost tracking state."""
        return {
            "total_cost_usd": round(self.total_cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "budget_remaining": round(max(0, self.max_cost_usd - self.total_cost_usd), 6),
            "budget_percent": (
                round(self.total_cost_usd / self.max_cost_usd * 100, 1)
                if self.max_cost_usd > 0
                else 0
            ),
            "total_tokens": self.total_tokens,
            "total_invocations": self.total_invocations,
            "elapsed_seconds": round(time.time() - self._start_time, 1),
            "cost_by_agent": dict(
                sorted(self.cost_by_agent.items(), key=lambda x: -x[1])
            ),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accumulate(self, agent_name: str, cost_usd: float, tokens: int) -> None:
        self.total_cost_usd += cost_usd
        self.total_tokens += tokens
        self.total_invocations += 1
        self.cost_by_agent[agent_name] = self.cost_by_agent.get(agent_name, 0) + cost_usd
        self.tokens_by_agent[agent_name] = self.tokens_by_agent.get(agent_name, 0) + tokens

    def _flush(self) -> None:
        if not self._artifacts_dir:
            return
        try:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._artifacts_dir / "cost_status.tmp"
            target = self._artifacts_dir / "cost_status.json"
            tmp.write_text(json.dumps(self.summary(), indent=2))
            tmp.rename(target)
        except Exception:
            pass


_current_cost_tracker: ContextVar[CostTracker | None] = ContextVar(
    "cost_tracker", default=None
)


def _track_cost(agent_name: str, response: object) -> None:
    """Record LLM cost to active CostTracker if available.

    Safe to call from any context — silently no-ops when there is no tracker
    or when the response lacks cost metrics.
    """
    try:
        tracker = _current_cost_tracker.get(None)
        if tracker and response and hasattr(response, "metrics") and response.metrics:
            cost = response.metrics.total_cost_usd or 0
            tokens = (response.metrics.input_tokens or 0) + (
                response.metrics.output_tokens or 0
            )
            if cost > 0:
                tracker.record_sync(agent_name, cost, tokens)
    except Exception:
        pass
