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
        CREDENTIAL_ENCRYPTION_KEY="custom-credential-key",
        MCP_API_KEY="custom-mcp-key",
    )

    _warn_on_insecure_defaults(settings)

    assert "One or more authentication secrets" not in caplog.text


def test_production_compose_requires_auth_secrets():
    compose = (REPO_ROOT / "compose.prod.yaml").read_text()

    assert "API_KEY: ${API_KEY:?API_KEY is required}" in compose
    assert "CHAT_API_KEY: ${API_KEY:?API_KEY is required}" in compose
    assert (
        "CREDENTIAL_ENCRYPTION_KEY: "
        "${CREDENTIAL_ENCRYPTION_KEY:?CREDENTIAL_ENCRYPTION_KEY is required}" in compose
    )
    assert "MCP_SERVERS: ${MCP_SERVERS:-[]}" in compose


def test_compose_forwards_legacy_mcp_settings_for_compatibility():
    for compose_file in ("compose.yaml", "compose.prod.yaml"):
        compose = (REPO_ROOT / compose_file).read_text()

        assert "MCP_API_KEY: ${MCP_API_KEY:-}" in compose
        assert "MCP_HTTP_URLS: ${MCP_HTTP_URLS:-}" in compose
        assert "MCP_SSE_URLS: ${MCP_SSE_URLS:-}" in compose


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
            CREDENTIAL_ENCRYPTION_KEY="custom-credential-key",
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


def test_sponsored_luna_defaults_are_disabled_and_safe():
    settings = Settings(SPONSORED_LUNA_ENABLED=False)

    assert settings.SPONSORED_LUNA_ENABLED is False
    assert settings.SPONSORED_OPENAI_API_KEY is None
    assert settings.SPONSORED_OPENAI_BASE_URL is None
    assert settings.SPONSORED_LUNA_UPSTREAM_MODEL == "gpt-5.6-luna"
    assert settings.SPONSORED_DAILY_USER_LIMIT == 5
    assert settings.SPONSORED_DAILY_GLOBAL_LIMIT is None
    assert settings.SPONSORED_MAX_OUTPUT_TOKENS == 1024
    assert settings.SPONSORED_REQUEST_LEASE_SECONDS == 600


def test_disabled_sponsored_luna_treats_blank_global_limit_as_unset(monkeypatch):
    monkeypatch.setenv("SPONSORED_DAILY_GLOBAL_LIMIT", "")

    settings = Settings(SPONSORED_LUNA_ENABLED=False)

    assert settings.SPONSORED_DAILY_GLOBAL_LIMIT is None


def test_sponsored_user_limit_cannot_exceed_five():
    with pytest.raises(ValueError, match="SPONSORED_DAILY_USER_LIMIT"):
        Settings(SPONSORED_DAILY_USER_LIMIT=6)


def test_enabled_sponsored_luna_requires_api_key():
    with pytest.raises(ValueError, match="SPONSORED_OPENAI_API_KEY") as error:
        Settings(
            SPONSORED_LUNA_ENABLED=True,
            SPONSORED_DAILY_GLOBAL_LIMIT=10,
        )

    assert "sponsored-secret" not in str(error.value)


def test_enabled_sponsored_luna_requires_positive_global_limit():
    with pytest.raises(ValueError, match="SPONSORED_DAILY_GLOBAL_LIMIT"):
        Settings(
            SPONSORED_LUNA_ENABLED=True,
            SPONSORED_OPENAI_API_KEY="sponsored-secret",
        )


def test_sponsored_luna_valid_key_uses_default_endpoint_and_masks_secret():
    settings = Settings(
        SPONSORED_LUNA_ENABLED=True,
        SPONSORED_OPENAI_API_KEY="sponsored-secret",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    assert settings.SPONSORED_OPENAI_API_KEY is not None
    assert str(settings.SPONSORED_OPENAI_API_KEY) == "**********"
    assert "sponsored-secret" not in repr(settings)


def test_sponsored_base_url_is_normalized():
    settings = Settings(
        SPONSORED_LUNA_ENABLED=True,
        SPONSORED_OPENAI_API_KEY="sponsored-secret",
        SPONSORED_OPENAI_BASE_URL=" HTTPS://Example.COM/v1/ ",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    assert settings.SPONSORED_OPENAI_BASE_URL == "https://example.com/v1"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://example.com/v1",
        "https:///v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?key=value",
        "https://example.com/v1#fragment",
    ],
)
def test_sponsored_base_url_rejects_unsafe_endpoints(base_url: str):
    with pytest.raises(ValueError, match="SPONSORED_OPENAI_BASE_URL"):
        Settings(SPONSORED_OPENAI_BASE_URL=base_url)


def test_compose_forwards_sponsored_luna_settings():
    expected_settings = (
        "SPONSORED_LUNA_ENABLED",
        "SPONSORED_OPENAI_API_KEY",
        "SPONSORED_OPENAI_BASE_URL",
        "SPONSORED_LUNA_UPSTREAM_MODEL",
        "SPONSORED_DAILY_USER_LIMIT",
        "SPONSORED_DAILY_GLOBAL_LIMIT",
        "SPONSORED_MAX_OUTPUT_TOKENS",
        "SPONSORED_REQUEST_LEASE_SECONDS",
    )

    for compose_file in ("compose.yaml", "compose.prod.yaml"):
        compose = (REPO_ROOT / compose_file).read_text()
        for setting_name in expected_settings:
            assert f"{setting_name}:" in compose
