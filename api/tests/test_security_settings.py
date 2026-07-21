import json
import os
import subprocess
from pathlib import Path

import pytest

from app.main import (
    _warn_on_insecure_defaults,
    lifespan,
    make_app,
)
from app.utils.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
type JsonValue = (
    dict[str, "JsonValue"] | list["JsonValue"] | str | int | float | bool | None
)

_COMPOSE_ENVIRONMENT = {
    "API_KEY": "compose-test-api-key",
    "AUTH_SECRET": "compose-test-auth-secret",
    "AUTH_URL": "http://localhost:3000",
    "CREDENTIAL_ENCRYPTION_KEY": "compose-test-credential-key",
    "GPU_API_URL": "http://gpu-api:8888",
    "LOG_LEVEL": "info",
    "PGADMIN_EMAIL": "compose-test@example.invalid",
    "PGADMIN_PASSWORD": "compose-test-password",
    "POSTGRES_DB": "compose_test",
    "POSTGRES_PASSWORD": "compose-test-password",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "compose_test",
    "TZ": "UTC",
    "WORKERS": "1",
}


def _compose_object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AssertionError(f"Expected {label} to be a JSON object")
    return value


def _render_compose(
    compose_file: str,
    extra_file: Path | None = None,
) -> dict[str, JsonValue]:
    command = ["docker", "compose", "-f", compose_file]
    if extra_file is not None:
        command.extend(("-f", str(extra_file)))
    command.extend(("config", "--format", "json"))

    environment = {"PATH": os.environ["PATH"], **_COMPOSE_ENVIRONMENT}
    result = subprocess.run(  # noqa: S603
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rendered: dict[str, JsonValue] = json.loads(result.stdout)
    return _compose_object(rendered, "Compose configuration")


def _assert_embedding_worker_contract(
    compose_file: str,
    *,
    extra_file: Path | None = None,
) -> None:
    rendered = _render_compose(compose_file, extra_file)
    services = _compose_object(rendered["services"], "services")
    api = _compose_object(services["api"], "api service")
    worker = _compose_object(services["embedding-worker"], "embedding worker service")

    assert worker["command"] == ["python", "-m", "app.embedding_worker"]
    depends_on = _compose_object(worker["depends_on"], "worker dependencies")
    database_dependency = _compose_object(depends_on["db"], "database dependency")
    assert database_dependency["condition"] == "service_healthy"
    assert worker["restart"] == "unless-stopped"
    assert worker["healthcheck"] == {"disable": True}
    assert "ports" not in worker
    assert "volumes" not in worker
    assert "working_dir" not in worker
    assert worker["networks"] == api["networks"]
    assert worker.get("pull_policy") == api.get("pull_policy")

    api_environment = _compose_object(api["environment"], "API environment")
    worker_environment = _compose_object(worker["environment"], "worker environment")
    expected_environment_keys = {
        "DATABASE_POOL_MAX_SIZE",
        "DATABASE_POOL_MIN_SIZE",
        "DATABASE_URL",
        "GPU_API_URL",
        "LOG_LEVEL",
        "TZ",
    }
    assert set(worker_environment) == expected_environment_keys
    assert {key: worker_environment[key] for key in expected_environment_keys} == {
        key: api_environment[key] for key in expected_environment_keys
    }

    if compose_file == "compose.yaml":
        api_build = _compose_object(api["build"], "API build")
        worker_build = _compose_object(worker["build"], "worker build")
        assert worker_build == api_build
        assert isinstance(worker_build["context"], str)
        assert worker_build["context"].endswith("/api")
        assert worker_build["dockerfile"] == "Dockerfile"
        assert worker["image"] == api["image"] == "finki-hub/chat-bot-api:latest"
        assert worker["pull_policy"] == api["pull_policy"] == "never"
    else:
        assert "build" not in worker
        assert (
            worker["image"] == api["image"] == ("ghcr.io/finki-hub/chat-bot-api:latest")
        )


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


def test_compose_defines_embedding_worker_runtime_contract():
    override = os.environ.get("COMPOSE_TEST_OVERRIDE")
    extra_file = Path(override) if override else None

    for compose_file in ("compose.yaml", "compose.prod.yaml"):
        _assert_embedding_worker_contract(compose_file, extra_file=extra_file)


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


def test_sponsored_model_defaults_are_disabled_and_safe():
    settings = Settings(SPONSORED_MODEL_ENABLED=False)

    assert settings.SPONSORED_MODEL_ENABLED is False
    assert settings.SPONSORED_MODEL_ID == "gpt-5.6-luna"
    assert settings.SPONSORED_MODEL_PROVIDER == "openai"
    assert settings.SPONSORED_MODEL_API_KEY is None
    assert settings.SPONSORED_MODEL_BASE_URL is None
    assert settings.SPONSORED_MODEL_UPSTREAM_MODEL == ""
    assert settings.SPONSORED_DAILY_USER_LIMIT == 5
    assert settings.SPONSORED_DAILY_GLOBAL_LIMIT is None
    assert settings.SPONSORED_MAX_OUTPUT_TOKENS == 1024
    assert settings.SPONSORED_REQUEST_LEASE_SECONDS == 600


def test_disabled_sponsored_model_treats_blank_global_limit_as_unset(monkeypatch):
    monkeypatch.setenv("SPONSORED_DAILY_GLOBAL_LIMIT", "")

    settings = Settings(SPONSORED_MODEL_ENABLED=False)

    assert settings.SPONSORED_DAILY_GLOBAL_LIMIT is None


def test_sponsored_user_limit_cannot_exceed_five():
    with pytest.raises(ValueError, match="SPONSORED_DAILY_USER_LIMIT"):
        Settings(SPONSORED_DAILY_USER_LIMIT=6)


def test_enabled_sponsored_model_requires_api_key():
    with pytest.raises(ValueError, match="SPONSORED_MODEL_API_KEY") as error:
        Settings(
            SPONSORED_MODEL_ENABLED=True,
            SPONSORED_DAILY_GLOBAL_LIMIT=10,
        )

    assert "sponsored-secret" not in str(error.value)


def test_enabled_sponsored_model_requires_positive_global_limit():
    with pytest.raises(ValueError, match="SPONSORED_DAILY_GLOBAL_LIMIT"):
        Settings(
            SPONSORED_MODEL_ENABLED=True,
            SPONSORED_MODEL_API_KEY="sponsored-secret",
        )


def test_enabled_sponsored_model_valid_key_uses_default_endpoint_and_masks_secret():
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY="sponsored-secret",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    assert settings.SPONSORED_MODEL_API_KEY is not None
    assert str(settings.SPONSORED_MODEL_API_KEY) == "**********"
    assert "sponsored-secret" not in repr(settings)


def test_sponsored_base_url_is_normalized():
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_API_KEY="sponsored-secret",
        SPONSORED_MODEL_BASE_URL=" HTTPS://Example.COM/v1/ ",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    assert settings.SPONSORED_MODEL_BASE_URL == "https://example.com/v1"


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
    with pytest.raises(ValueError, match="SPONSORED_MODEL_BASE_URL"):
        Settings(SPONSORED_MODEL_BASE_URL=base_url)


def test_sponsored_model_rejects_ollama_model_tags_as_out_of_catalog():
    assert "SPONSORED_MODEL_ID" in Settings.model_fields

    with pytest.raises(ValueError, match="SPONSORED_MODEL_ID"):
        Settings(
            SPONSORED_MODEL_ENABLED=True,
            SPONSORED_MODEL_ID="qwen3:14b-q4_K_M",
            SPONSORED_MODEL_PROVIDER="ollama",
            SPONSORED_MODEL_API_KEY="sponsored-secret",
            SPONSORED_DAILY_GLOBAL_LIMIT=10,
        )


def test_sponsored_model_rejects_provider_mismatch():
    assert "SPONSORED_MODEL_PROVIDER" in Settings.model_fields

    with pytest.raises(ValueError, match="SPONSORED_MODEL_PROVIDER"):
        Settings(
            SPONSORED_MODEL_ENABLED=True,
            SPONSORED_MODEL_ID="gpt-5.6-luna",
            SPONSORED_MODEL_PROVIDER="google",
            SPONSORED_MODEL_API_KEY="sponsored-secret",
            SPONSORED_DAILY_GLOBAL_LIMIT=10,
        )


def test_sponsored_model_reports_derived_provider_mismatch_before_configured_ollama_rejection():
    assert "SPONSORED_MODEL_PROVIDER" in Settings.model_fields

    with pytest.raises(
        ValueError,
        match="SPONSORED_MODEL_PROVIDER must match provider derived from SPONSORED_MODEL_ID",
    ):
        Settings(
            SPONSORED_MODEL_ENABLED=True,
            SPONSORED_MODEL_ID="gpt-5.6-luna",
            SPONSORED_MODEL_PROVIDER="ollama",
            SPONSORED_MODEL_API_KEY="sponsored-secret",
            SPONSORED_DAILY_GLOBAL_LIMIT=10,
        )


def test_sponsored_model_rejects_out_of_catalog_id():
    assert "SPONSORED_MODEL_ID" in Settings.model_fields

    with pytest.raises(ValueError, match="SPONSORED_MODEL_ID"):
        Settings(
            SPONSORED_MODEL_ENABLED=True,
            SPONSORED_MODEL_ID="gpt-4.1",
            SPONSORED_MODEL_PROVIDER="openai",
            SPONSORED_MODEL_API_KEY="sponsored-secret",
            SPONSORED_DAILY_GLOBAL_LIMIT=10,
        )


def test_sponsored_model_accepts_non_luna_profile():
    settings = Settings(
        SPONSORED_MODEL_ENABLED=True,
        SPONSORED_MODEL_ID="gemini-3.5-flash",
        SPONSORED_MODEL_PROVIDER="google",
        SPONSORED_MODEL_API_KEY="sponsored-secret",
        SPONSORED_DAILY_GLOBAL_LIMIT=10,
    )

    assert settings.SPONSORED_MODEL_ID == "gemini-3.5-flash"
    assert settings.SPONSORED_MODEL_PROVIDER == "google"


def test_sponsored_legacy_env_names_are_not_recognized():
    assert "SPONSORED_LUNA_ENABLED" not in Settings.model_fields
    assert "SPONSORED_OPENAI_API_KEY" not in Settings.model_fields
    assert "SPONSORED_OPENAI_BASE_URL" not in Settings.model_fields
    assert "SPONSORED_LUNA_UPSTREAM_MODEL" not in Settings.model_fields


def test_compose_forwards_sponsored_model_settings():
    expected_settings = (
        "SPONSORED_MODEL_ENABLED",
        "SPONSORED_MODEL_ID",
        "SPONSORED_MODEL_PROVIDER",
        "SPONSORED_MODEL_API_KEY",
        "SPONSORED_MODEL_BASE_URL",
        "SPONSORED_MODEL_UPSTREAM_MODEL",
        "SPONSORED_DAILY_USER_LIMIT",
        "SPONSORED_DAILY_GLOBAL_LIMIT",
        "SPONSORED_MAX_OUTPUT_TOKENS",
        "SPONSORED_REQUEST_LEASE_SECONDS",
    )

    for compose_file in ("compose.yaml", "compose.prod.yaml"):
        compose = (REPO_ROOT / compose_file).read_text()
        for setting_name in expected_settings:
            assert f"{setting_name}:" in compose
