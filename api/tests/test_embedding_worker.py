from __future__ import annotations

from dataclasses import dataclass, field

import anyio

import app.embedding_worker as worker
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.embedding_worker import WorkerDependencies, run_worker
from app.embedding_worker_drain import DirtyDrainReport
from tests.embedding_worker_test_support import WorkerListener, no_delay


@dataclass
class ResourceState:
    events: list[str] = field(default_factory=list)
    client_closed: bool = False
    drain_count: int = 0
    drain_targets: list[EmbeddingCorpus | None] = field(default_factory=list)
    drain_event: anyio.Event = field(default_factory=anyio.Event)

    def open_client(self) -> None:
        self.events.append("client-open")

    async def close_client(self) -> None:
        self.client_closed = True
        self.events.append("client-close")
        await no_delay(0)


@dataclass
class FakeDatabase:
    state: ResourceState
    disconnected: bool = False

    async def init(self) -> None:
        self.state.events.append("database-init")
        await no_delay(0)

    async def run_migrations(self) -> None:
        self.state.events.append("migrations")
        await no_delay(0)

    async def disconnect(self) -> None:
        self.disconnected = True
        self.state.events.append("database-close")
        await no_delay(0)


@dataclass
class MigrationBlockedDatabase(FakeDatabase):
    migration_started: anyio.Event = field(default_factory=anyio.Event)
    migration_blocker: anyio.Event = field(default_factory=anyio.Event)

    async def run_migrations(self) -> None:
        self.state.events.append("migrations")
        self.migration_started.set()
        await self.migration_blocker.wait()


class FakeListener(WorkerListener):
    def __init__(self, state: ResourceState) -> None:
        super().__init__(
            on_listen=lambda: state.events.append("listen"),
            on_close=lambda: state.events.append("listener-close"),
        )


class ListenerFactory:
    def __init__(self, listeners: list[FakeListener]) -> None:
        self._listeners = listeners
        self._index = 0

    async def __call__(self, _database_url: str):
        listener = self._listeners[self._index]
        self._index += 1
        await no_delay(0)
        return listener


async def _drain(
    state: ResourceState,
    target: EmbeddingCorpus | None = None,
) -> DirtyDrainReport:
    state.events.append("drain")
    state.drain_count += 1
    state.drain_targets.append(target)
    state.drain_event.set()
    await no_delay(0)
    return DirtyDrainReport(0, 0, 0, 0)


async def _wait_for_drains(state: ResourceState, expected: int) -> None:
    while state.drain_count < expected:
        event = state.drain_event
        await event.wait()
        if event is state.drain_event:
            state.drain_event = anyio.Event()


def _dependencies(
    state: ResourceState,
    database: FakeDatabase,
    listeners: list[FakeListener],
) -> WorkerDependencies:
    async def drain(target: EmbeddingCorpus | None) -> DirtyDrainReport:
        return await _drain(state, target)

    return WorkerDependencies(
        database=database,
        database_url="postgresql://worker-test",
        connect_listener=ListenerFactory(listeners),
        drain=drain,
        open_client=state.open_client,
        close_client=state.close_client,
        reconnect_delay=no_delay,
    )


def test_worker_migrates_and_drains_before_listening() -> None:
    async def run() -> None:
        # Given: an empty worker runtime with a dedicated listener connection.
        state = ResourceState()
        database = FakeDatabase(state)
        listener = FakeListener(state)

        # When: the worker starts and reaches its LISTEN wait state.
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                run_worker,
                _dependencies(state, database, [listener]),
            )
            await listener.listening.wait()

            await _wait_for_drains(state, 1)

            # Then: migrations precede the scan; LISTEN is registered before it to close
            # the notification race, and the worker only blocks after the drain returns.
            assert state.events[:5] == [
                "database-init",
                "migrations",
                "client-open",
                "listen",
                "drain",
            ]
            task_group.cancel_scope.cancel()

        assert listener.closed
        assert database.disconnected
        assert state.client_closed

    anyio.run(run)


def test_worker_closes_initialized_resources_when_migration_is_cancelled() -> None:
    async def run() -> None:
        # Given: an initialized database whose migration remains in progress.
        state = ResourceState()
        database = MigrationBlockedDatabase(state)

        # When: worker cancellation interrupts the migration await point.
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                run_worker,
                _dependencies(state, database, []),
            )
            await database.migration_started.wait()
            task_group.cancel_scope.cancel()

        # Then: process-owned resources are closed despite partial startup.
        assert database.disconnected
        assert state.client_closed

    anyio.run(run)


def test_worker_main_configures_logging_from_log_level(monkeypatch) -> None:
    configured_levels: list[str] = []

    def setup_logging(level: str) -> None:
        configured_levels.append(level)

    def run(_func, _dependencies) -> None:
        return None

    # Given: the standalone worker entrypoint receives a non-default log level.
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(worker, "setup_logging", setup_logging, raising=False)
    monkeypatch.setattr(worker.anyio, "run", run)

    # When: the module entrypoint starts.
    worker.main()

    # Then: worker logging uses the same project configuration path as the API.
    assert configured_levels == ["DEBUG"]


def test_worker_coalesces_notification_bursts_and_stays_idle_without_polling() -> None:
    async def run() -> None:
        # Given: a started worker which completed its startup drain.
        state = ResourceState()
        database = FakeDatabase(state)
        listener = FakeListener(state)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                run_worker,
                _dependencies(state, database, [listener]),
            )
            await listener.listening.wait()
            await _wait_for_drains(state, 1)

            # When: several payload-agnostic notifications arrive before the wake drain.
            listener.notify()
            listener.notify()
            listener.notify()
            with anyio.fail_after(1):
                await _wait_for_drains(state, 2)

            # Then: the burst becomes one drain and an idle observation adds no scans.
            assert state.drain_count == 2
            with anyio.move_on_after(0.05) as idle_window:
                await state.drain_event.wait()
            assert idle_window.cancelled_caught
            assert state.drain_count == 2
            task_group.cancel_scope.cancel()

    anyio.run(run)


def test_worker_reconnects_and_runs_a_new_full_scan_after_listener_termination() -> (
    None
):
    async def run() -> None:
        # Given: a listener that terminates and a replacement dedicated connection.
        state = ResourceState()
        database = FakeDatabase(state)
        first = FakeListener(state)
        second = FakeListener(state)
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(
                run_worker,
                _dependencies(state, database, [first, second]),
            )
            await first.listening.wait()

            # When: PostgreSQL terminates the first listener backend.
            first.terminate()
            await second.listening.wait()
            await _wait_for_drains(state, 2)

            # Then: reconnect registration closes the race and is followed by a full drain.
            assert state.events.count("drain") == 2
            assert first.closed
            assert state.events.index("listen", 1) < state.events.index("drain", 1)
            task_group.cancel_scope.cancel()

    anyio.run(run)
