from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

_LOGGER = logging.getLogger("decider_service")
request_id_context: ContextVar[str | None] = ContextVar(
    "timing_request_id", default=None
)


@dataclass
class StageTiming:
    outcome: str = "success"
    details: dict[str, int | float] = field(default_factory=dict)


@contextmanager
def measure_stage(stage: str, **details: int | float) -> Iterator[StageTiming]:
    """Measure request wall time without recording inputs or exception text."""
    timing = StageTiming(details=details)
    request_id = request_id_context.get()
    if request_id is None:
        yield timing
        return
    started = perf_counter()
    try:
        yield timing
    except asyncio.CancelledError:
        timing.outcome = "cancelled"
        raise
    except BaseException:
        timing.outcome = "error"
        raise
    finally:
        duration_ms = (perf_counter() - started) * 1000
        numeric_fields = "".join(
            f" {name}={value}" for name, value in timing.details.items()
        )
        _LOGGER.info(
            "request_timing request_id=%s stage=%s duration_ms=%.3f outcome=%s%s",
            request_id,
            stage,
            duration_ms,
            timing.outcome,
            numeric_fields,
            extra={
                "request_id": request_id,
                "stage": stage,
                "duration_ms": duration_ms,
                "outcome": timing.outcome,
                **timing.details,
            },
        )
