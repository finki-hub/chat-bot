import asyncio
import logging

from app.data.connection import Database
from app.data.retrieval_gaps import RetrievalGap, insert_retrieval_gap

logger = logging.getLogger(__name__)


class RetrievalGapWrites:
    """Best-effort background inserts that never delay or fail an answer."""

    def __init__(self) -> None:
        self.pending: set[asyncio.Task[None]] = set()
        self.accepting = True

    def schedule(self, db: Database, gap: RetrievalGap) -> None:
        if not self.accepting:
            return
        task = asyncio.create_task(_store(db, gap))
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def drain(self, *, grace_seconds: float = 5.0) -> None:
        self.accepting = False
        if not self.pending:
            return
        tasks = tuple(self.pending)
        try:
            await asyncio.wait(tasks, timeout=grace_seconds)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.pending.difference_update(tasks)


async def _store(db: Database, gap: RetrievalGap) -> None:
    try:
        await insert_retrieval_gap(db, gap)
    # ruff: ignore[BLE001] -- a failed diagnostic write must never surface to the answer
    except Exception as exc:
        logger.warning(
            "Retrieval gap write failed response_id=%s error_type=%s",
            gap.response_id,
            type(exc).__name__,
        )
