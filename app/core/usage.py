"""
Per-execution LLM usage accounting.

One agent run makes several LLM calls, and not all of them are visible from
the execution loop: `search_knowledge` triggers a query expansion and a
grounded generation inside the RAG pipeline, several layers down. Threading
a usage object through every call signature would couple the RAG code to
billing.

Instead the accumulator lives in a ContextVar. `chat_completion` records
into whichever accumulator is active, so a caller wraps a run in
`track_usage()` and gets the total — including anything nested. ContextVars
are per-task and per-thread, so concurrent requests never share one.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

from app.config import MODEL_PRICING
from app.logger import get_logger

logger = get_logger(__name__)

_unpriced_models_warned: set[str] = set()


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Usage:
    """Totals for one tracked scope, broken down by model."""

    by_model: dict[str, ModelUsage] = field(default_factory=dict)
    latency_ms: int = 0

    @property
    def prompt_tokens(self) -> int:
        return sum(m.prompt_tokens for m in self.by_model.values())

    @property
    def completion_tokens(self) -> int:
        return sum(m.completion_tokens for m in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def llm_calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def cost_usd(self) -> float | None:
        """
        Cost in USD, or None when any model involved has no configured
        price.

        None rather than 0.0 on purpose: a zero would read as "this run was
        free", which is the wrong thing to show on a billing page. Prices
        are configuration — see config.MODEL_PRICING.
        """
        total = 0.0
        for model, usage in self.by_model.items():
            price = MODEL_PRICING.get(model)
            if price is None:
                _warn_unpriced(model)
                return None
            total += (
                usage.prompt_tokens * price["input"]
                + usage.completion_tokens * price["output"]
            ) / 1_000_000
        return round(total, 8)

    def add(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        entry = self.by_model.setdefault(model, ModelUsage())
        entry.prompt_tokens += prompt_tokens
        entry.completion_tokens += completion_tokens
        entry.calls += 1

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "by_model": {
                model: {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                    "calls": u.calls,
                }
                for model, u in self.by_model.items()
            },
        }


_current_usage: ContextVar[Usage | None] = ContextVar(
    "current_usage", default=None
)


@contextmanager
def track_usage() -> Iterator[Usage]:
    """
    Collects usage for everything that happens inside the block.

    Nesting is allowed; the innermost scope wins, and the token is reset on
    exit so an exception cannot leak the accumulator into the next request.
    """
    usage = Usage()
    token = _current_usage.set(usage)
    started = time.perf_counter()
    try:
        yield usage
    finally:
        usage.latency_ms = int((time.perf_counter() - started) * 1000)
        _current_usage.reset(token)


def record(model: str, response) -> None:
    """
    Records one completion against the active scope. A no-op when nothing
    is tracking, so library code can call it unconditionally.
    """
    usage = _current_usage.get()
    if usage is None:
        return

    reported = getattr(response, "usage", None)
    prompt_tokens = getattr(reported, "prompt_tokens", 0) or 0
    completion_tokens = getattr(reported, "completion_tokens", 0) or 0

    usage.add(model, prompt_tokens, completion_tokens)


def _warn_unpriced(model: str) -> None:
    """Logs an unpriced model once, not on every request."""
    if model in _unpriced_models_warned:
        return
    _unpriced_models_warned.add(model)
    logger.warning(
        f"No price configured for model | model={model} — cost will be "
        f"reported as null. Set MODEL_PRICING to enable cost accounting."
    )
