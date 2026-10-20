import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from anyio import CapacityLimiter, to_thread

logger = logging.getLogger(__name__)


class Lane(StrEnum):
    EMBED = "embed"
    RERANK = "rerank"


# One worker per model: concurrent forward passes of the same model add no throughput
# on a single GPU, they only make every in-flight request finish late together. An
# embed and a rerank may still overlap: a ~30 ms embed waiting behind a ~450 ms rerank
# would cost more than sharing the GPU briefly.
_limiters: dict[Lane, CapacityLimiter] = {lane: CapacityLimiter(1) for lane in Lane}


@dataclass(frozen=True, slots=True)
class LaneRun[T]:
    value: T
    queue_ms: float
    run_ms: float


async def run_in_lane[**P, T](
    lane: Lane,
    function: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> LaneRun[T]:
    """Run blocking inference on the lane's single worker, timing queue and run apart."""
    limiter = _limiters[lane]
    stats = limiter.statistics()
    pending = stats.borrowed_tokens + stats.tasks_waiting
    submitted = perf_counter()
    started: float | None = None
    finished: float | None = None

    def call() -> T:
        nonlocal started, finished
        started = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            finished = perf_counter()

    # Without abandon_on_cancel, a cancelled caller still waits for a running job, so
    # the limiter and the timings below never pretend the GPU is free while it is busy.
    try:
        value = await to_thread.run_sync(call, limiter=limiter)
    finally:
        start = started if started is not None else perf_counter()
        end = finished if finished is not None else start
        queue_ms = round((start - submitted) * 1000, 1)
        run_ms = round((end - start) * 1000, 1)
        logger.info(
            "gpu.lane %s",
            json.dumps(
                {
                    "lane": lane.value,
                    "pending": pending,
                    "queue_ms": queue_ms,
                    "run_ms": run_ms,
                    "ran": started is not None,
                },
            ),
        )
    return LaneRun(value=value, queue_ms=queue_ms, run_ms=run_ms)
