"""Structured JSONL tracing: one line per event, one file per day.

This is the observability layer. Every model request, tool call, and error
lands here with token counts and cost so a session can be audited after the
fact without re-running anything.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.config import CONFIG, LOG_DIR
from src.utils.guardrails import redact

_MAX_FIELD = 4000


def _clip(value: Any) -> Any:
    """Redact secrets and clip long strings so traces stay readable."""
    if isinstance(value, str):
        value = redact(value)
        return value if len(value) <= _MAX_FIELD else value[:_MAX_FIELD] + f"...<clipped {len(value) - _MAX_FIELD} chars>"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value[:50]]
    return value


class Tracer:
    """Append-only trace for a single agent session."""

    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.path = LOG_DIR / f"trace-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        self.turn = 0
        self.total_cost = 0.0
        self.total_input = 0
        self.total_output = 0
        self.total_cached = 0

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": self.session_id,
            "turn": self.turn,
            "event": event,
            **{k: _clip(v) for k, v in fields.items()},
        }
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            # Tracing must never take the agent down.
            pass

    def record_usage(self, usage) -> float:
        cost = CONFIG.cost(usage)
        self.total_cost += cost
        self.total_input += getattr(usage, "input_tokens", 0) or 0
        self.total_output += getattr(usage, "output_tokens", 0) or 0
        self.total_cached += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.emit(
            "model_response",
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_read=getattr(usage, "cache_read_input_tokens", 0),
            cache_write=getattr(usage, "cache_creation_input_tokens", 0),
            cost_usd=round(cost, 6),
        )
        return cost

    def tool_call(self, name: str, tool_input: Any, result: str, ok: bool, elapsed: float) -> None:
        self.emit(
            "tool_call",
            tool=name,
            input=tool_input,
            ok=ok,
            elapsed_ms=round(elapsed * 1000, 1),
            result_preview=result[:600] if isinstance(result, str) else result,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "turns": self.turn,
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "cached_tokens": self.total_cached,
            "cost_usd": round(self.total_cost, 4),
            "trace_file": str(self.path),
        }


class Timer:
    """Tiny context manager for measuring tool latency."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.elapsed = time.perf_counter() - self._start
