from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.
    """

    APP_TITLE: str = "GPU API"
    APP_DESCRIPTION: str = (
        "API providing GPU-accelerated embeddings and chat capabilities."
    )
    API_VERSION: str = "1.0.0"

    LOG_LEVEL: str = "INFO"

    PRELOAD_BGEM3: bool = True

    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    POSTHOG_KEY: str = ""
    POSTHOG_HOST: str = "https://eu.i.posthog.com"

    ALLOWED_ORIGINS: list[str] = ["*"]
    EXPOSE_HEADERS: list[str] = ["*"]

    HOST: str = "0.0.0.0"  # noqa: S104
    PORT: int = 8888
