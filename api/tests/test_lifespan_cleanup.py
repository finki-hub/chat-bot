from collections.abc import Callable

import pytest

from app import main
from app.utils.settings import Settings


class FakeDatabase:
    events: list[str]

    def __init__(self, _dsn: str, _min_size: int, _max_size: int) -> None:
        self.events = []

    async def init(self) -> None:
        self.events.append("database_init")

    async def disconnect(self) -> None:
        self.events.append("database_disconnect")


def _patch_database(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], FakeDatabase]:
    databases: list[FakeDatabase] = []

    class CapturingDatabase(FakeDatabase):
        def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
            super().__init__(dsn, min_size, max_size)
            databases.append(self)

    monkeypatch.setattr(main, "Database", CapturingDatabase)
    return lambda: databases[0]


@pytest.mark.anyio
async def test_api_lifespan_cleans_up_after_body_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _patch_database(monkeypatch)
    http_events: list[str] = []

    monkeypatch.setattr(
        main,
        "init_http_client",
        lambda: http_events.append("http_init"),
    )

    async def close_http_client() -> None:
        http_events.append("http_close")

    monkeypatch.setattr(main, "close_http_client", close_http_client)
    app = main.make_app(Settings())

    with pytest.raises(RuntimeError, match="body failed"):
        async with main.lifespan(app):
            raise RuntimeError("body failed")

    assert http_events == ["http_init", "http_close"]
    assert database().events == ["database_init", "database_disconnect"]


@pytest.mark.anyio
async def test_api_lifespan_disconnects_database_when_http_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _patch_database(monkeypatch)

    def fail_http_startup() -> None:
        raise RuntimeError("HTTP startup failed")

    monkeypatch.setattr(main, "init_http_client", fail_http_startup)
    app = main.make_app(Settings())

    with pytest.raises(RuntimeError, match="HTTP startup failed"):
        async with main.lifespan(app):
            pass

    assert database().events == ["database_init", "database_disconnect"]


@pytest.mark.anyio
async def test_api_lifespan_disconnects_database_when_http_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _patch_database(monkeypatch)
    monkeypatch.setattr(main, "init_http_client", lambda: None)

    async def fail_http_cleanup() -> None:
        raise RuntimeError("HTTP cleanup failed")

    monkeypatch.setattr(main, "close_http_client", fail_http_cleanup)
    app = main.make_app(Settings())

    with pytest.raises(RuntimeError, match="HTTP cleanup failed"):
        async with main.lifespan(app):
            pass

    assert database().events == ["database_init", "database_disconnect"]
