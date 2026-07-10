from ipaddress import ip_address
from typing import Annotated, Final, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings

_API_KEY_DEFAULT: Final[str] = "your_api_key_here"
_MCP_API_KEY_DEFAULT: Final[str] = "SystemPass"
_CREDENTIAL_ENCRYPTION_KEY_DEFAULT: Final[str] = "your_byok_encryption_key_here"
McpTransport = Literal["streamable_http", "sse"]
NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _is_insecure_secret(value: str, default: str) -> bool:
    normalized = value.strip()
    return normalized in ("", default)


def _canonical_byok_base_url(value: str) -> str | None:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() != "https" or parsed.hostname is None:
        return None
    if parsed.username or parsed.password or parsed.fragment:
        return None
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        return None
    try:
        address = ip_address(hostname)
    except ValueError:
        pass
    else:
        if not address.is_global:
            return None
    try:
        parsed_port = parsed.port
    except ValueError:
        return None
    port = f":{parsed_port}" if parsed_port is not None else ""
    path = parsed.path.rstrip("/")
    return f"https://{hostname}{port}{path}"


class McpServerSettings(BaseModel):
    """Connection and tool exposure settings for one MCP server."""

    model_config = ConfigDict(frozen=True)

    name: NonBlankString
    url: NonBlankString
    transport: McpTransport
    api_key: str = ""
    allowed_tools: tuple[str, ...] = ()
    blocked_tools: tuple[str, ...] = ()


class Settings(BaseSettings):
    """
    Application settings.
    """

    APP_TITLE: str = "API"
    APP_DESCRIPTION: str = "API managing questions, links, and LLM interactions."
    API_VERSION: str = "1.0.0"

    LOG_LEVEL: str = "INFO"
    TZ: str = "Europe/Skopje"

    GPU_API_URL: str = "http://gpu-api:8888"
    DIPLOMAS_API_URL: str = "https://diplomski-api.finki-hub.com/diplomas"
    STAFF_API_URL: str = "https://assets.finki-hub.com/staff.json"
    STAFF_CACHE_TTL: int = 3600
    # Minimum cross-encoder score (sentence-transformers CrossEncoder.predict applies a
    # sigmoid for num_labels=1 rerankers like bge-reranker-v2-m3, so scores are 0..1) a
    # candidate must clear to enter the final context. Tune per reranker distribution.
    RERANKER_MIN_SCORE: float = 0.1
    SOURCE_RERANKER_MIN_SCORE: float = 0.3
    MCP_HTTP_URLS: str = ""
    MCP_SSE_URLS: str = ""
    MCP_SERVERS: list[McpServerSettings] = Field(default_factory=list)
    MCP_TOOLS_TTL: int = 3600
    MCP_API_KEY: str = _MCP_API_KEY_DEFAULT

    API_KEY: str = _API_KEY_DEFAULT
    CREDENTIAL_ENCRYPTION_KEY: str = _CREDENTIAL_ENCRYPTION_KEY_DEFAULT
    DATABASE_URL: str = "postgresql://user:password@host:port/db"
    DATABASE_POOL_MIN_SIZE: int = Field(default=1, ge=0)
    DATABASE_POOL_MAX_SIZE: int = Field(default=10, ge=1)

    OLLAMA_URL: str = "http://ollama:11434"
    BYOK_ALLOWED_BASE_URLS: str = ""

    POSTHOG_KEY: str = ""
    POSTHOG_HOST: str = "https://eu.i.posthog.com"

    CHAT_HISTORY_MAX_TURNS: int = 10

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8880

    @model_validator(mode="after")
    def validate_database_pool_sizes(self) -> Self:
        if self.DATABASE_POOL_MIN_SIZE > self.DATABASE_POOL_MAX_SIZE:
            msg = "DATABASE_POOL_MIN_SIZE must be <= DATABASE_POOL_MAX_SIZE"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_mcp_server_names(self) -> Self:
        names = [server.name for server in self.MCP_SERVERS]
        if len(names) != len(set(names)):
            msg = "MCP_SERVERS entries must have unique names"
            raise ValueError(msg)
        return self

    @field_validator("MCP_HTTP_URLS", "MCP_SSE_URLS", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: object) -> str:
        """Normalise the value to a plain string — empty or comma-separated URLs."""
        return str(v).strip() if v else ""

    @field_validator("BYOK_ALLOWED_BASE_URLS", mode="before")
    @classmethod
    def parse_byok_base_urls(cls, v: object) -> str:
        """Normalise the BYOK custom endpoint allowlist to comma-separated URLs."""
        return str(v).strip() if v else ""

    def insecure_secret_names(self) -> list[str]:
        insecure_names: list[str] = []
        if _is_insecure_secret(self.API_KEY, _API_KEY_DEFAULT):
            insecure_names.append("API_KEY")
        if _is_insecure_secret(
            self.CREDENTIAL_ENCRYPTION_KEY,
            _CREDENTIAL_ENCRYPTION_KEY_DEFAULT,
        ):
            insecure_names.append("CREDENTIAL_ENCRYPTION_KEY")
        if (self.MCP_HTTP_URLS or self.MCP_SSE_URLS) and _is_insecure_secret(
            self.MCP_API_KEY,
            _MCP_API_KEY_DEFAULT,
        ):
            insecure_names.append("MCP_API_KEY")
        insecure_names.extend(
            f"MCP_SERVERS.{server.name}.api_key"
            for server in self.MCP_SERVERS
            if server.api_key
            and _is_insecure_secret(
                server.api_key,
                _MCP_API_KEY_DEFAULT,
            )
        )
        return insecure_names

    def mcp_http_url_list(self) -> list[str]:
        return [u for u in self.MCP_HTTP_URLS.split(",") if u]

    def mcp_sse_url_list(self) -> list[str]:
        return [u for u in self.MCP_SSE_URLS.split(",") if u]

    def byok_allowed_base_url_list(self) -> list[str]:
        urls: list[str] = []
        for url in self.BYOK_ALLOWED_BASE_URLS.split(","):
            canonical = _canonical_byok_base_url(url)
            if canonical is not None:
                urls.append(canonical)
        return urls

    def is_byok_base_url_allowed(self, value: str) -> bool:
        canonical = _canonical_byok_base_url(value)
        return canonical is not None and canonical in self.byok_allowed_base_url_list()

    def mcp_server_configs(self) -> list[McpServerSettings]:
        if self.MCP_SERVERS:
            return self.MCP_SERVERS

        return [
            *[
                McpServerSettings(
                    name=url,
                    url=url,
                    transport="streamable_http",
                    api_key=self.MCP_API_KEY,
                )
                for url in self.mcp_http_url_list()
            ],
            *[
                McpServerSettings(
                    name=url,
                    url=url,
                    transport="sse",
                    api_key=self.MCP_API_KEY,
                )
                for url in self.mcp_sse_url_list()
            ],
        ]
