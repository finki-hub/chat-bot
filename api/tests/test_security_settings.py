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
    assert "MCP_API_KEY: ${MCP_API_KEY:?MCP_API_KEY is required}" in compose


@pytest.mark.anyio
async def test_lifespan_warns_for_insecure_app_settings(monkeypatch, caplog):
    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
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
async def test_lifespan_uses_app_settings_database_url(monkeypatch):
    captured_dsns: list[str] = []

    class FakeDatabase:
        def __init__(self, dsn: str) -> None:
            captured_dsns.append(dsn)

        async def init(self) -> None:
            return None

        async def disconnect(self) -> None:
            return None

    monkeypatch.setattr("app.main.Database", FakeDatabase)
    app = make_app(
        Settings(
            API_KEY="custom-api-key",
            DATABASE_URL="postgresql://custom-user:custom-pass@custom-host/custom-db",
            MCP_API_KEY="custom-mcp-key",
        ),
    )

    async with lifespan(app):
        pass

    assert captured_dsns == [
        "postgresql://custom-user:custom-pass@custom-host/custom-db",
    ]
