import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token


def _round(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None


class RequestTimings:
    """Per-request stage timings in milliseconds.

    Shared (via a context var) across the retrieval coroutines that run under
    asyncio.gather; they interleave on one thread, so the plain dict needs no locking.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._thinking_t0: float | None = None
        self.spans: dict[str, float] = {}
        self.ttft_ms: float | None = None
        self.total_ms: float | None = None
        self.thinking_ms: float | None = None
        self.candidate_count: int | None = None
        self.top_distance: float | None = None
        self.retrieval_ids: list[str] = []
        self.reranker_score_max: float | None = None
        self.reranker_score_min: float | None = None
        self.reranker_above_threshold: int | None = None

    def record(self, name: str, elapsed_ms: float) -> None:
        self.spans[name] = self.spans.get(name, 0.0) + elapsed_ms

    def mark_ttft(self) -> None:
        if self.ttft_ms is None:
            self.ttft_ms = (time.perf_counter() - self._t0) * 1000.0

    def mark_thinking(self) -> None:
        if self._thinking_t0 is None:
            self._thinking_t0 = time.perf_counter()

    def mark_answer(self) -> None:
        if self.thinking_ms is None and self._thinking_t0 is not None:
            self.thinking_ms = (time.perf_counter() - self._thinking_t0) * 1000.0

    def mark_total(self) -> None:
        self.total_ms = (time.perf_counter() - self._t0) * 1000.0

    def as_record(self) -> dict[str, object]:
        return {
            "ttft_ms": _round(self.ttft_ms),
            "total_ms": _round(self.total_ms),
            "thinking_ms": _round(self.thinking_ms),
            "candidate_count": self.candidate_count,
            "top_distance": _round(self.top_distance, 4),
            "spans": {name: round(ms, 1) for name, ms in self.spans.items()},
        }


_current: ContextVar[RequestTimings | None] = ContextVar(
    "request_timings",
    default=None,
)


def start_request_timings() -> tuple[RequestTimings, Token]:
    """Open a timing scope for the current request."""
    timings = RequestTimings()
    token = _current.set(timings)
    return timings, token


def reset_request_timings(token: Token) -> None:
    _current.reset(token)


def record_retrieval_shape(candidate_count: int, top_distance: float | None) -> None:
    timings = _current.get()
    if timings is not None:
        timings.candidate_count = candidate_count
        timings.top_distance = top_distance


def record_retrieval_ids(ids: list[str]) -> None:
    timings = _current.get()
    if timings is not None:
        timings.retrieval_ids = ids


def record_reranker_scores(scores: list[float], above_threshold: int) -> None:
    timings = _current.get()
    if timings is None or not scores:
        return
    timings.reranker_score_max = max(scores)
    timings.reranker_score_min = min(scores)
    timings.reranker_above_threshold = above_threshold


@contextmanager
def timed(name: str) -> Iterator[None]:
    """Time the wrapped block into the active request's timings.

    A no-op outside a request scope, so the retrieval helpers stay callable from the
    eval harness, which runs them without one.
    """
    timings = _current.get()
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings.record(name, (time.perf_counter() - start) * 1000.0)
