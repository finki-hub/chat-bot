from __future__ import annotations

from dataclasses import dataclass, field

import anyio

from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.embedding_worker import WorkerDependencies, run_worker
from app.embedding_worker_drain import DirtyDrainReport
from tests.embedding_worker_test_support import (
    NoopWorkerDatabase,
    WorkerListener,
    no_delay,
)
from tests.embedding_worker_test_support import close_client as noop_close_client


@dataclass
class NotificationState:
    drain_targets: list[EmbeddingCorpus | None] = field(default_factory=list)
    drain_event: anyio.Event = field(default_factory=anyio.Event)

    async def close_client(self) -> None:
        await noop_close_client()


NotificationListener = WorkerListener


def _dependencies(
    state: NotificationState,
    listener: NotificationListener,
) -> WorkerDependencies:
    async def connect_listener(_database_url: str) -> NotificationListener:
        await no_delay(0)
        return listener

    async def drain(target: EmbeddingCorpus | None) -> DirtyDrainReport:
        state.drain_targets.append(target)
        state.drain_event.set()
        await no_delay(0)
        return DirtyDrainReport(0, 0, 0, 0)

    return WorkerDependencies(
        database=NoopWorkerDatabase(),
        database_url="postgresql://worker-test",
        connect_listener=connect_listener,
        drain=drain,
        open_client=lambda: None,
        close_client=state.close_client,
        reconnect_delay=no_delay,
    )


async def _wait_for_drains(state: NotificationState, expected: int) -> None:
    while len(state.drain_targets) < expected:
        event = state.drain_event
        await event.wait()
        if event is state.drain_event:
            state.drain_event = anyio.Event()


def test_worker_uses_notification_payload_as_corpus_hint() -> None:
    async def run() -> None:
        # Given: a worker that completed its startup full scan.
        state = NotificationState()
        listener = NotificationListener()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_worker, _dependencies(state, listener))
            await listener.listening.wait()
            await _wait_for_drains(state, 1)

            # When: PostgreSQL reports one dirty question row.
            listener.notify("question:00000000-0000-0000-0000-000000000421")
            with anyio.fail_after(1):
                await _wait_for_drains(state, 2)

            # Then: the notification drain targets only the hinted corpus.
            assert state.drain_targets == [None, EmbeddingCorpus.QUESTION]
            task_group.cancel_scope.cancel()

    anyio.run(run)


def test_worker_uses_full_scan_for_administrative_wake_payload() -> None:
    async def run() -> None:
        # Given: a worker that completed its startup full scan.
        state = NotificationState()
        listener = NotificationListener()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_worker, _dependencies(state, listener))
            await listener.listening.wait()
            await _wait_for_drains(state, 1)

            # When: an administrative wake payload arrives.
            listener.notify("wake")
            with anyio.fail_after(1):
                await _wait_for_drains(state, 2)

            # Then: the worker preserves the durable full-scan recovery path.
            assert state.drain_targets == [None, None]
            task_group.cancel_scope.cancel()

    anyio.run(run)


def test_worker_uses_full_scan_for_malformed_notification_payload() -> None:
    async def run() -> None:
        # Given: a worker that completed its startup full scan.
        state = NotificationState()
        listener = NotificationListener()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_worker, _dependencies(state, listener))
            await listener.listening.wait()
            await _wait_for_drains(state, 1)

            # When: PostgreSQL reports a malformed dirty payload.
            listener.notify("question:not-a-uuid")
            with anyio.fail_after(1):
                await _wait_for_drains(state, 2)

            # Then: the worker falls back to full-scan recovery.
            assert state.drain_targets == [None, None]
            task_group.cancel_scope.cancel()

    anyio.run(run)


def test_worker_uses_full_scan_for_mixed_corpus_notification_burst() -> None:
    async def run() -> None:
        # Given: a worker that completed its startup full scan.
        state = NotificationState()
        listener = NotificationListener()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_worker, _dependencies(state, listener))
            await listener.listening.wait()
            await _wait_for_drains(state, 1)

            # When: a burst reports dirty rows from more than one corpus.
            listener.notify("question:00000000-0000-0000-0000-000000000422")
            listener.notify("chunk:00000000-0000-0000-0000-000000000423")
            with anyio.fail_after(1):
                await _wait_for_drains(state, 2)

            # Then: the coalesced drain covers every durable corpus.
            assert state.drain_targets == [None, None]
            task_group.cancel_scope.cancel()

    anyio.run(run)
