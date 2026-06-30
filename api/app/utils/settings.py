from typing import Final

from pydantic import field_validator
from pydantic_settings import BaseSettings

_PRODUCTION_ENVIRONMENTS: Final[frozenset[str]] = frozenset({"prod", "production"})


class Settings(BaseSettings):
    """
    Application settings.
    """

    APP_TITLE: str = "API"
    APP_DESCRIPTION: str = "API managing questions, links, and LLM interactions."
    API_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    LOG_LEVEL: str = "INFO"
    TZ: str = "Europe/Skopje"

    GPU_API_URL: str = "http://gpu-api:8888"
    DIPLOMAS_API_URL: str = "https://diplomski-api.finki-hub.com/diplomas"
    # Minimum cross-encoder score (sentence-transformers CrossEncoder.predict applies a
    # sigmoid for num_labels=1 rerankers like bge-reranker-v2-m3, so scores are 0..1) a
    # candidate must clear to enter the final context. Tune per reranker distribution.
    RERANKER_MIN_SCORE: float = 0.1
    MCP_HTTP_URLS: str = ""
    MCP_SSE_URLS: str = ""
    MCP_TOOLS_TTL: int = 3600
    MCP_API_KEY: str = "SystemPass"

    API_KEY: str = "your_api_key_here"
    DATABASE_URL: str = "postgresql://user:password@host:port/db"

    OPENAI_API_KEY: str = "your_openai_api_key_here"
    GOOGLE_API_KEY: str = "your_google_api_key_here"
    ANTHROPIC_API_KEY: str = "your_anthropic_api_key_here"

    OLLAMA_URL: str = "http://ollama:11434"
    OPENAI_BASE_URL: str = ""
    GOOGLE_BASE_URL: str = ""
    ANTHROPIC_BASE_URL: str = ""

    POSTHOG_KEY: str = ""
    POSTHOG_HOST: str = "https://eu.i.posthog.com"

    CHAT_HISTORY_MAX_TURNS: int = 10

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8880

    @field_validator("MCP_HTTP_URLS", "MCP_SSE_URLS", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: object) -> str:
        """Normalise the value to a plain string — empty or comma-separated URLs."""
        return str(v).strip() if v else ""

    def is_production(self) -> bool:
        return self.ENVIRONMENT.casefold() in _PRODUCTION_ENVIRONMENTS

    def insecure_secret_names(self) -> list[str]:
        fields = type(self).model_fields
        default_values = (self.API_KEY, self.MCP_API_KEY)
        return sorted(
            name
            for name, field in fields.items()
            if isinstance(field.default, str)
            and field.default in default_values
            and getattr(self, name) == field.default
        )

    def mcp_http_url_list(self) -> list[str]:
        return [u for u in self.MCP_HTTP_URLS.split(",") if u]

    def mcp_sse_url_list(self) -> list[str]:
        return [u for u in self.MCP_SSE_URLS.split(",") if u]
