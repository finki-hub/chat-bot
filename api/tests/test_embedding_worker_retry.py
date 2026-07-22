from __future__ import annotations

from collections.abc import Awaitable, Callable

import anyio

import app.embedding_worker as worker
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.embedding_worker_drain import DirtyDrainReport
from tests.embedding_worker_test_support import (
    NoopWorkerDatabase,
    WorkerListener,
    close_client,
    no_delay,
)

RetryListener = WorkerListener


def _dependencies(
    drain: Callable[[EmbeddingCorpus | None], Awaitable[DirtyDrainReport]],
    listener: RetryListener,
) -> worker.WorkerDependencies:
    async def connect_listener(_database_url: str) -> RetryListener:
        await no_delay(0)
        return listener

    return worker.WorkerDependencies(
        database=NoopWorkerDatabase(),
        database_url="postgresql://worker-test",
        connect_listener=connect_listener,
        drain=drain,
        open_client=lambda: None,
        close_client=close_client,
        reconnect_delay=no_delay,
    )


def test_listener_retries_failed_drain_without_notification(monkeypatch) -> None:
    async def run() -> None:
        # Given: a failed dirty drain followed by a successful drain and no notification.
        listener = RetryListener()
        calls = 0
        retry_completed = anyio.Event()

        async def drain(_target: EmbeddingCorpus | None) -> DirtyDrainReport:
            nonlocal calls
            calls += 1
            if calls == 1:
                await no_delay(0)
                return DirtyDrainReport(1, 0, 1, 0)
            retry_completed.set()
            await no_delay(0)
            return DirtyDrainReport(0, 0, 0, 0)

        # When: the session starts with a retry delay that does not wait in the test.
        monkeypatch.setattr(worker.anyio, "sleep", no_delay)
        monkeypatch.setattr(worker, "_FAILED_DRAIN_RETRY_MIN_SECONDS", 0.001)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                worker.run_worker,
                _dependencies(drain, listener),
            )
            await listener.ready.wait()
            with anyio.fail_after(0.1):
                await retry_completed.wait()

            # Then: failed durable work is retried without an external wakeup.
            assert calls == 2
            task_group.cancel_scope.cancel()

    anyio.run(run)


def test_listener_preserves_failed_full_scan_retry_after_targeted_notification(
    monkeypatch,
) -> None:
    async def run() -> None:
        # Given: a failed full scan and a later targeted notification during backoff.
        listener = RetryListener()
        calls = 0
        targets: list[EmbeddingCorpus | None] = []
        first_failure = anyio.Event()
        retry_completed = anyio.Event()

        async def drain(target: EmbeddingCorpus | None) -> DirtyDrainReport:
            nonlocal calls
            calls += 1
            targets.append(target)
            if calls == 1:
                first_failure.set()
                await no_delay(0)
                return DirtyDrainReport(1, 0, 1, 0)
            retry_completed.set()
            await no_delay(0)
            return DirtyDrainReport(0, 0, 0, 0)

        # When: a targeted wake arrives while failed durable work is awaiting retry.
        monkeypatch.setattr(worker.anyio, "sleep", no_delay)
        monkeypatch.setattr(worker, "_FAILED_DRAIN_RETRY_MIN_SECONDS", 30.0)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                worker.run_worker,
                _dependencies(drain, listener),
            )
            await listener.ready.wait()
            await first_failure.wait()
            listener.notify("chunk:00000000-0000-0000-0000-000000000431")
            with anyio.fail_after(0.1):
                await retry_completed.wait()

            # Then: the retry still uses full scan to include the failed scope.
            assert targets == [None, None]
            task_group.cancel_scope.cancel()

    anyio.run(run)


def test_listener_cancels_while_waiting_to_retry_persistent_failure(
    monkeypatch,
) -> None:
    async def run() -> None:
        # Given: persistent failed work and a long bounded retry wait.
        listener = RetryListener()
        calls = 0
        first_failure = anyio.Event()

        async def drain(_target: EmbeddingCorpus | None) -> DirtyDrainReport:
            nonlocal calls
            calls += 1
            first_failure.set()
            await no_delay(0)
            return DirtyDrainReport(1, 0, 1, 0)

        # When: cancellation arrives after the first failed drain enters backoff.
        monkeypatch.setattr(worker, "_FAILED_DRAIN_RETRY_MIN_SECONDS", 30.0)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                worker.run_worker,
                _dependencies(drain, listener),
            )
            await listener.ready.wait()
            await first_failure.wait()
            task_group.cancel_scope.cancel()

        # Then: cancellation is responsive and does not spin another drain.
        assert calls == 1

    anyio.run(run)
