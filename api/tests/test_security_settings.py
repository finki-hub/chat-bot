from pathlib import Path

import pytest

from app.main import (
    _warn_on_insecure_defaults,
    lifespan,
    make_app,
)
from app.utils.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_warn_on_insecure_defaults_warns_for_default_secrets(caplog):
    _warn_on_insecure_defaults(Settings())

    assert "One or more authentication secrets" in caplog.text


def test_warn_on_insecure_defaults_warns_for_blank_secrets(caplog):
    settings = Settings(
        API_KEY="",
        MCP_API_KEY="   ",
    )

    _warn_on_insecure_defaults(settings)

    assert "One or more authentication secrets" in caplog.text


def test_warn_on_insecure_defaults_warns_for_whitespace_default_secrets(caplog):
    settings = Settings(
        API_KEY=" your_api_key_here ",
        MCP_API_KEY=" SystemPass ",
    )

    _warn_on_insecure_defaults(settings)

    assert "One or more authentication secrets" in caplog.text


def test_warn_on_insecure_defaults_allows_custom_secrets(caplog):
    settings = Settings(
        API_KEY="custom-api-key",
        MCP_API_KEY="custom-mcp-key",
    )

    _warn_on_insecure_defaults(settings)

    assert "One or more authentication secrets" not in caplog.text


def test_production_compose_requires_auth_secrets():
    compose = (REPO_ROOT / "compose.prod.yaml").read_text()

    assert "API_KEY: ${API_KEY:?API_KEY is required}" in compose
    assert "CHAT_API_KEY: ${API_KEY:?API_KEY is required}" in compose
    assert "MCP_SERVERS: ${MCP_SERVERS:-[]}" in compose


@pytest.mark.anyio
async def test_lifespan_warns_for_insecure_app_settings(monkeypatch, caplog):
    class FakeDatabase:
        def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
            return None

        async def init(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr("app.main.Database", FakeDatabase)
    app = make_app(Settings())

    async with lifespan(app):
        pass

    assert "One or more authentication secrets" in caplog.text


@pytest.mark.anyio
async def test_lifespan_uses_app_settings_database_config(monkeypatch):
    captured_config: list[tuple[str, int, int]] = []

    class FakeDatabase:
        def __init__(self, dsn: str, min_size: int, max_size: int) -> None:
            captured_config.append((dsn, min_size, max_size))

        async def init(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr("app.main.Database", FakeDatabase)
    app = make_app(
        Settings(
            API_KEY="custom-api-key",
            DATABASE_POOL_MAX_SIZE=12,
            DATABASE_POOL_MIN_SIZE=2,
            DATABASE_URL="postgresql://custom-user:custom-pass@custom-host/custom-db",
            MCP_API_KEY="custom-mcp-key",
        ),
    )

    async with lifespan(app):
        pass

    assert captured_config == [
        ("postgresql://custom-user:custom-pass@custom-host/custom-db", 2, 12),
    ]


def test_database_pool_min_size_cannot_exceed_max_size():
    with pytest.raises(ValueError, match="DATABASE_POOL_MIN_SIZE"):
        Settings(
            DATABASE_POOL_MAX_SIZE=1,
            DATABASE_POOL_MIN_SIZE=2,
        )
