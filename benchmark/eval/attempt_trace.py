"""Helpers for attempt-level provider-call tracing."""

from __future__ import annotations

from typing import Any


def attempt_event(
    *,
    attempt: int,
    status: str,
    provider_calls: int = 1,
    detail: str | None = None,
) -> dict[str, Any]:
    event = {
        "type": "provider_attempt",
        "attempt": attempt,
        "status": status,
        "provider_calls": provider_calls,
    }
    if detail:
        event["detail"] = detail
    return event


def retries_from_attempts(attempt_count: int) -> int:
    """Retries are failed/preceding attempts before the final one."""
    return max(0, attempt_count - 1)
