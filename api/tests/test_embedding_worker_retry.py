from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio

import app.embedding_worker as worker
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.embedding_worker import ListenerConnection, WorkerDependencies
from app.embedding_worker_drain import DirtyDrainReport


@dataclass
class RetryDatabase:
    async def init(self) -> None:
        return None

    async def run_migrations(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None


class RetryListener(ListenerConnection):
    def __init__(self) -> None:
        self.ready = anyio.Event()
        self._notification_callback: Callable[..., None] | None = None
        self._termination_callback: Callable[..., None] | None = None

    async def add_listener(
        self,
        channel: str,
        callback: Callable[..., None],
    ) -> None:
        assert channel == "embedding_dirty"
        self._notification_callback = callback

    def add_termination_listener(self, callback: Callable[..., None]) -> None:
        self._termination_callback = callback

    async def execute(self, query: str) -> str:
        assert query == "LISTEN embedding_dirty"
        self.ready.set()
        return "LISTEN"

    async def close(self) -> None:
        return None

    def notify(self, payload: str) -> None:
        assert self._notification_callback is not None
        self._notification_callback(self, 1, "embedding_dirty", payload)


async def _close_client() -> None:
    return None


async def _no_delay(_seconds: float) -> None:
    return None


def _dependencies(
    drain: Callable[[EmbeddingCorpus | None], Awaitable[DirtyDrainReport]],
    listener: RetryListener,
) -> WorkerDependencies:
    async def connect_listener(_database_url: str) -> RetryListener:
        return listener

    return WorkerDependencies(
        database=RetryDatabase(),
        database_url="postgresql://worker-test",
        connect_listener=connect_listener,
        drain=drain,
        open_client=lambda: None,
        close_client=_close_client,
        reconnect_delay=_no_delay,
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
                return DirtyDrainReport(1, 0, 1, 0)
            retry_completed.set()
            return DirtyDrainReport(0, 0, 0, 0)

        # When: the session starts with a retry delay that does not wait in the test.
        monkeypatch.setattr(worker.anyio, "sleep", _no_delay)
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
                return DirtyDrainReport(1, 0, 1, 0)
            retry_completed.set()
            return DirtyDrainReport(0, 0, 0, 0)

        # When: a targeted wake arrives while failed durable work is awaiting retry.
        monkeypatch.setattr(worker.anyio, "sleep", _no_delay)
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
