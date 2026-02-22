from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    """

    APP_TITLE: str = "API"
    APP_DESCRIPTION: str = "API managing questions, links, and LLM interactions."
    API_VERSION: str = "1.0.0"

    LOG_LEVEL: str = "INFO"

    GPU_API_URL: str = "http://gpu-api:8888"
    MCP_HTTP_URLS: str = ""
    MCP_SSE_URLS: str = ""

    API_KEY: str = "your_api_key_here"
    DATABASE_URL: str = "postgresql+asyncpg://user:password@host:port/db"

    OLLAMA_URL: str = "http://ollama:11434"
    OPENAI_API_KEY: str = "your_openai_api_key_here"
    GOOGLE_API_KEY: str = "your_google_api_key_here"

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8880

    @field_validator("MCP_HTTP_URLS", "MCP_SSE_URLS", mode="before")
    @classmethod
    def parse_comma_separated(cls, v: object) -> str:
        """Normalise the value to a plain string — empty or comma-separated URLs."""
        return str(v).strip() if v else ""

    def mcp_http_url_list(self) -> list[str]:
        return [u for u in self.MCP_HTTP_URLS.split(",") if u]

    def mcp_sse_url_list(self) -> list[str]:
        return [u for u in self.MCP_SSE_URLS.split(",") if u]
