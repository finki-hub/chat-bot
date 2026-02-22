import httpx

_http_client: httpx.AsyncClient | None = None


def init_http_client() -> None:
    """Initialize the shared HTTP client. Call once during app lifespan startup."""
    global _http_client  # noqa: PLW0603
    _http_client = httpx.AsyncClient(timeout=30.0)


async def close_http_client() -> None:
    """Close the shared HTTP client. Call once during app lifespan shutdown."""
    global _http_client  # noqa: PLW0603
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def get_http_client() -> httpx.AsyncClient:
    """Return the shared HTTP client (must be initialized first)."""
    if _http_client is None:
        raise RuntimeError("HTTP client has not been initialized")
    return _http_client
