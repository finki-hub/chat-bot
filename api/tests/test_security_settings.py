from pathlib import Path

import pytest

import app.main as app_main
from app.main import (
    SecurityConfigurationError,
    _validate_security_config,
    lifespan,
    make_app,
)
from app.utils.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_validate_security_config_allows_development_defaults():
    _validate_security_config(Settings(ENVIRONMENT="development"))


def test_validate_security_config_rejects_production_default_secrets():
    with pytest.raises(SecurityConfigurationError, match="API_KEY, MCP_API_KEY"):
        _validate_security_config(Settings(ENVIRONMENT="production"))


def test_validate_security_config_rejects_production_blank_secrets():
    settings = Settings(
        API_KEY="",
        ENVIRONMENT="production",
        MCP_API_KEY="   ",
    )

    with pytest.raises(SecurityConfigurationError, match="API_KEY, MCP_API_KEY"):
        _validate_security_config(settings)


def test_validate_security_config_rejects_production_whitespace_default_secrets():
    settings = Settings(
        API_KEY=" your_api_key_here ",
        ENVIRONMENT="production",
        MCP_API_KEY=" SystemPass ",
    )

    with pytest.raises(SecurityConfigurationError, match="API_KEY, MCP_API_KEY"):
        _validate_security_config(settings)


def test_validate_security_config_allows_production_custom_secrets():
    settings = Settings(
        API_KEY="custom-api-key",
        ENVIRONMENT="production",
        MCP_API_KEY="custom-mcp-key",
    )

    _validate_security_config(settings)


def test_production_compose_enables_production_environment():
    compose = (REPO_ROOT / "compose.prod.yaml").read_text()

    assert "ENVIRONMENT: ${ENVIRONMENT:-production}" in compose


def test_production_compose_requires_auth_secrets():
    compose = (REPO_ROOT / "compose.prod.yaml").read_text()

    assert "API_KEY: ${API_KEY:?API_KEY is required}" in compose
    assert "CHAT_API_KEY: ${API_KEY:?API_KEY is required}" in compose
    assert "MCP_API_KEY: ${MCP_API_KEY:?MCP_API_KEY is required}" in compose


@pytest.mark.anyio
async def test_lifespan_validates_app_settings():
    app = make_app(Settings(ENVIRONMENT="production"))

    with pytest.raises(SecurityConfigurationError, match="API_KEY, MCP_API_KEY"):
        async with lifespan(app):
            pass


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

    monkeypatch.setattr(app_main, "Database", FakeDatabase)
    app = make_app(
        Settings(
            API_KEY="custom-api-key",
            DATABASE_URL="postgresql://custom-user:custom-pass@custom-host/custom-db",
            ENVIRONMENT="production",
            MCP_API_KEY="custom-mcp-key",
        ),
    )

    async with lifespan(app):
        pass

    assert captured_dsns == [
        "postgresql://custom-user:custom-pass@custom-host/custom-db",
    ]
