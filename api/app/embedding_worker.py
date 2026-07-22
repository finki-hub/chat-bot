import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import anyio
from asyncpg import InterfaceError, PostgresError, connect

from app.data.connection import Database
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.embedding_worker_drain import DirtyDrainReport, drain_dirty_embeddings
from app.utils.http_client import close_http_client, init_http_client
from app.utils.logger import setup_logging
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_CHANNEL = "embedding_dirty"
_RECONNECT_MIN_SECONDS = 0.5
_RECONNECT_MAX_SECONDS = 30.0
_FAILED_DRAIN_RETRY_MIN_SECONDS = 0.5
_FAILED_DRAIN_RETRY_MAX_SECONDS = 30.0


class WorkerDatabase(Protocol):
    """The pooled database lifecycle used by the worker process."""

    async def init(self) -> None:
        raise NotImplementedError

    async def run_migrations(self) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError


class ListenerConnection(Protocol):
    """The dedicated non-pooled connection behavior the worker needs."""

    async def add_listener(
        self,
        channel: str,
        callback: Callable[..., None],
    ) -> None:
        raise NotImplementedError

    def add_termination_listener(self, callback: Callable[..., None]) -> None:
        raise NotImplementedError

    async def execute(self, query: str) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError


type ListenerConnector = Callable[[str], Awaitable[ListenerConnection]]
type WorkerDrain = Callable[[EmbeddingCorpus | None], Awaitable[DirtyDrainReport]]
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


@dataclass(frozen=True, slots=True)
class DrainRequest:
    """A coalesced durable drain request observed from LISTEN state."""

    corpus: EmbeddingCorpus | None
    listener_terminated: bool


@dataclass(slots=True)
class ListenerSignals:
    """Coalesce LISTEN callbacks while retaining termination as durable state."""

    event: anyio.Event = field(default_factory=anyio.Event)
    generation: int = 0
    listener_terminated: bool = False
    requested_corpus: EmbeddingCorpus | None = None
    full_scan_requested: bool = False

    def notification[ConnectionReference, Payload](
        self,
        _connection: ConnectionReference,
        _process_id: int,
        _channel: str,
        payload: Payload,
    ) -> None:
        corpus = _corpus_from_payload(payload)
        if corpus is None:
            self.full_scan_requested = True
            self.requested_corpus = None
        elif not self.full_scan_requested:
            if self.requested_corpus is None or self.requested_corpus is corpus:
                self.requested_corpus = corpus
            else:
                self.full_scan_requested = True
                self.requested_corpus = None
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

    def consume_request(self) -> DrainRequest:
        corpus = None if self.full_scan_requested else self.requested_corpus
        self.full_scan_requested = False
        self.requested_corpus = None
        return DrainRequest(corpus, self.listener_terminated)

    async def wait_after(self, generation: int) -> DrainRequest:
        if self.listener_terminated:
            return self.consume_request()
        if generation != self.generation:
            self.event = anyio.Event()
            return self.consume_request()
        event = self.event
        await event.wait()
        self.event = anyio.Event()
        return self.consume_request()

    async def wait_for_retry(self, delay: float) -> DrainRequest:
        """Wait for a wakeup or capped retry delay after durable work failed."""
        event = self.event
        with anyio.move_on_after(delay):
            await event.wait()
        self.event = anyio.Event()
        return self.consume_request()


def _corpus_from_payload(payload: object) -> EmbeddingCorpus | None:
    if not isinstance(payload, str):
        return None
    corpus_name, separator, row_id = payload.partition(":")
    if separator != ":":
        return None
    try:
        UUID(row_id)
        return EmbeddingCorpus(corpus_name)
    except ValueError:
        return None


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
    drain_corpus: EmbeddingCorpus | None = None
    while True:
        generation = signals.generation
        report = await dependencies.drain(drain_corpus)
        if report.failed_batches or report.invalid_batches:
            logger.warning(
                "embedding_worker.drain retrying failed=%d invalid=%d delay=%.1f",
                report.failed_batches,
                report.invalid_batches,
                retry_delay,
            )
            request = await signals.wait_for_retry(retry_delay)
            if request.listener_terminated:
                logger.warning("embedding_worker.listener terminated")
                return
            drain_corpus = None
            retry_delay = min(retry_delay * 2, _FAILED_DRAIN_RETRY_MAX_SECONDS)
            continue

        retry_delay = _FAILED_DRAIN_RETRY_MIN_SECONDS
        request = await signals.wait_after(generation)
        if request.listener_terminated:
            logger.warning("embedding_worker.listener terminated")
            return
        drain_corpus = request.corpus


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
        await _run_listener_forever(dependencies)
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
        drain=lambda corpus: drain_dirty_embeddings(database, corpus=corpus),
        open_client=init_http_client,
        close_client=close_http_client,
        reconnect_delay=anyio.sleep,
    )


def main() -> None:
    """Start the reconnect-safe embedding invalidation worker command."""
    setup_logging(level=Settings().LOG_LEVEL)
    anyio.run(run_worker, default_dependencies())


if __name__ == "__main__":
    main()
