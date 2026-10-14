import asyncio
import contextvars
import json
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from time import perf_counter

logger = logging.getLogger(__name__)


class Lane(StrEnum):
    EMBED = "embed"
    RERANK = "rerank"


# One worker per model: concurrent forward passes on a single GPU add no throughput,
# they only make every in-flight request finish late together and stack activation
# memory. A FIFO lane per model keeps the GPU busy while letting earlier requests
# finish first, and an embed never queues behind a rerank's CPU-side work.
_executors: dict[Lane, ThreadPoolExecutor] = {
    lane: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"gpu-{lane}")
    for lane in Lane
}
_pending: dict[Lane, int] = dict.fromkeys(Lane, 0)


async def run_in_lane[**P, T](
    lane: Lane,
    function: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run blocking inference on the lane's single worker and log its queue wait."""
    context = contextvars.copy_context()
    submitted = perf_counter()
    started: float | None = None

    def _call() -> T:
        nonlocal started
        started = perf_counter()
        return context.run(function, *args, **kwargs)

    pending_at_submit = _pending[lane]
    _pending[lane] += 1
    try:
        return await asyncio.get_running_loop().run_in_executor(
            _executors[lane],
            _call,
        )
    finally:
        _pending[lane] -= 1
        finished = perf_counter()
        start = finished if started is None else started
        logger.info(
            "gpu.lane %s",
            json.dumps(
                {
                    "lane": lane.value,
                    "pending": pending_at_submit,
                    "queue_ms": round((start - submitted) * 1000, 1),
                    "run_ms": round((finished - start) * 1000, 1),
                },
            ),
        )
