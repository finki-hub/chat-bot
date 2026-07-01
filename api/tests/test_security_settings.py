from pathlib import Path

import pytest

from app.main import SecurityConfigurationError, _validate_security_config
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
