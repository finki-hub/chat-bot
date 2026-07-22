from __future__ import annotations

from collections.abc import Callable

import anyio
from anyio.lowlevel import checkpoint

from app.embedding_worker import ListenerConnection


class WorkerListener(ListenerConnection):
    def __init__(
        self,
        *,
        on_listen: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.closed = False
        self.listening = anyio.Event()
        self.ready = self.listening
        self._notification_callback: Callable[..., None] | None = None
        self._termination_callback: Callable[..., None] | None = None
        self._on_listen = on_listen
        self._on_close = on_close

    async def add_listener(
        self,
        channel: str,
        callback: Callable[..., None],
    ) -> None:
        assert channel == "embedding_dirty"
        self._notification_callback = callback
        await checkpoint()

    def add_termination_listener(self, callback: Callable[..., None]) -> None:
        self._termination_callback = callback

    async def execute(self, query: str) -> str:
        assert query == "LISTEN embedding_dirty"
        if self._on_listen is not None:
            self._on_listen()
        self.listening.set()
        await checkpoint()
        return "LISTEN"

    async def close(self) -> None:
        self.closed = True
        if self._on_close is not None:
            self._on_close()
        await checkpoint()

    def notify(self, payload: str = "wake") -> None:
        assert self._notification_callback is not None
        self._notification_callback(self, 1, "embedding_dirty", payload)

    def terminate(self) -> None:
        assert self._termination_callback is not None
        self._termination_callback(self)


async def no_delay(_seconds: float) -> None:
    await checkpoint()


async def close_client() -> None:
    await checkpoint()


class NoopWorkerDatabase:
    async def init(self) -> None:
        await checkpoint()

    async def run_migrations(self) -> None:
        await checkpoint()

    async def disconnect(self) -> None:
        await checkpoint()
