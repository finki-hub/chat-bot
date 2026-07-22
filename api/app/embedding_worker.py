import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import anyio
from asyncpg import InterfaceError, PostgresError, connect

from app.data.connection import Database
from app.embedding_worker_drain import DirtyDrainReport, drain_dirty_embeddings
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_CHANNEL = "embedding_dirty"
_RECONNECT_MIN_SECONDS = 0.5
_RECONNECT_MAX_SECONDS = 30.0
_FAILED_DRAIN_RETRY_MIN_SECONDS = 0.5
_FAILED_DRAIN_RETRY_MAX_SECONDS = 30.0


class WorkerDatabase(Protocol):
    """The pooled database lifecycle used by the worker process."""

    async def init(self) -> None: ...

    async def run_migrations(self) -> None: ...

    async def disconnect(self) -> None: ...


class ListenerConnection(Protocol):
    """The dedicated non-pooled connection behavior the worker needs."""

    async def add_listener(
        self,
        channel: str,
        callback: Callable[..., None],
    ) -> None: ...

    def add_termination_listener(self, callback: Callable[..., None]) -> None: ...

    async def execute(self, query: str) -> str: ...

    async def close(self) -> None: ...


type ListenerConnector = Callable[[str], Awaitable[ListenerConnection]]
type WorkerDrain = Callable[[], Awaitable[DirtyDrainReport]]
type ReconnectDelay = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class WorkerDependencies:
    """Explicit process-bound resources and behavior seams for the worker."""

    database: WorkerDatabase
    database_url: str
    connect_listener: ListenerConnector
    drain: WorkerDrain
    open_client: Callable[[], None]
    close_client: Callable[[], Awaitable[None]]
    reconnect_delay: ReconnectDelay


@dataclass(slots=True)
class ListenerSignals:
    """Coalesce LISTEN callbacks while retaining termination as durable state."""

    event: anyio.Event = field(default_factory=anyio.Event)
    generation: int = 0
    listener_terminated: bool = False

    def notification[ConnectionReference, Payload](
        self,
        _connection: ConnectionReference,
        _process_id: int,
        _channel: str,
        _payload: Payload,
    ) -> None:
        """Record a payload-agnostic wakeup from asyncpg's callback."""
        self.generation += 1
        self.event.set()

    def termination[ConnectionReference](
        self,
        _connection: ConnectionReference,
    ) -> None:
        """Unblock the session when asyncpg reports a lost listener connection."""
        self.listener_terminated = True
        self.generation += 1
        self.event.set()

    async def wait_after(self, generation: int) -> bool:
        """Wait for a newer wakeup and return whether it was listener termination."""
        if self.listener_terminated:
            return self.listener_terminated
        if generation != self.generation:
            self.event = anyio.Event()
            return False
        event = self.event
        await event.wait()
        if event is self.event:
            self.event = anyio.Event()
        return self.listener_terminated

    async def wait_for_retry(self, delay: float) -> bool:
        """Wait for a wakeup or capped retry delay after durable work failed."""
        event = self.event
        with anyio.move_on_after(delay):
            await event.wait()
        if event is self.event:
            self.event = anyio.Event()
        return self.listener_terminated


async def _connect_listener(database_url: str) -> ListenerConnection:
    """Open the dedicated asyncpg connection that owns LISTEN state."""
    return await connect(database_url)


async def _close_listener(listener: ListenerConnection) -> None:
    """Close the dedicated listener even when structured cancellation is active."""
    with anyio.CancelScope(shield=True):
        await listener.close()


async def _run_listener_session(
    dependencies: WorkerDependencies,
    listener: ListenerConnection,
) -> None:
    """Drain once per observed dirty-state transition until the listener terminates."""
    signals = ListenerSignals()
    await listener.add_listener(_CHANNEL, signals.notification)
    listener.add_termination_listener(signals.termination)
    await listener.execute(f"LISTEN {_CHANNEL}")
    logger.info("embedding_worker.listener connected")

    retry_delay = _FAILED_DRAIN_RETRY_MIN_SECONDS
    while True:
        generation = signals.generation
        report = await dependencies.drain()
        if report.failed_batches or report.invalid_batches:
            logger.warning(
                "embedding_worker.drain retrying failed=%d invalid=%d delay=%.1f",
                report.failed_batches,
                report.invalid_batches,
                retry_delay,
            )
            if await signals.wait_for_retry(retry_delay):
                logger.warning("embedding_worker.listener terminated")
                return
            retry_delay = min(retry_delay * 2, _FAILED_DRAIN_RETRY_MAX_SECONDS)
            continue

        retry_delay = _FAILED_DRAIN_RETRY_MIN_SECONDS
        if await signals.wait_after(generation):
            logger.warning("embedding_worker.listener terminated")
            return


async def _run_listener_forever(dependencies: WorkerDependencies) -> None:
    """Reconnect only after listener failures, resetting backoff after each connection."""
    delay = _RECONNECT_MIN_SECONDS
    while True:
        listener: ListenerConnection | None = None
        try:
            listener = await dependencies.connect_listener(dependencies.database_url)
            await _run_listener_session(dependencies, listener)
            delay = _RECONNECT_MIN_SECONDS
        except (InterfaceError, OSError, PostgresError) as error:
            logger.warning(
                "embedding_worker.listener connection_failed error_type=%s",
                type(error).__name__,
            )
        finally:
            if listener is not None:
                await _close_listener(listener)

        await dependencies.reconnect_delay(delay)
        delay = min(delay * 2, _RECONNECT_MAX_SECONDS)


async def run_worker(dependencies: WorkerDependencies) -> None:
    """Run migrations, own process resources, and serve durable dirty-work wakeups."""
    try:
        await dependencies.database.init()
        await dependencies.database.run_migrations()
        dependencies.open_client()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_run_listener_forever, dependencies)
    finally:
        with anyio.CancelScope(shield=True):
            await dependencies.close_client()
            await dependencies.database.disconnect()


def default_dependencies() -> WorkerDependencies:
    """Build production dependencies from the application settings."""
    settings = Settings()
    database = Database(
        settings.DATABASE_URL,
        min_size=settings.DATABASE_POOL_MIN_SIZE,
        max_size=settings.DATABASE_POOL_MAX_SIZE,
    )
    return WorkerDependencies(
        database=database,
        database_url=settings.DATABASE_URL,
        connect_listener=_connect_listener,
        drain=lambda: drain_dirty_embeddings(database),
        open_client=init_http_client,
        close_client=close_http_client,
        reconnect_delay=anyio.sleep,
    )


def main() -> None:
    """Start the reconnect-safe embedding invalidation worker command."""
    anyio.run(run_worker, default_dependencies())


if __name__ == "__main__":
    main()
