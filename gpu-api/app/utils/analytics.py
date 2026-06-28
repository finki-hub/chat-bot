import logging

from posthog import Posthog

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_SERVICE = "chat-bot-gpu-api"


class _State:
    client: Posthog | None = None


_state = _State()


def init_analytics(settings: Settings) -> None:
    """Construct the PostHog client; a no-op when ``POSTHOG_KEY`` is unset (dev/CI/tests).

    Called from the app lifespan, which runs per-worker after gunicorn forks, so the
    posthog-python background flush thread is created in the worker that uses it.
    """
    if not settings.POSTHOG_KEY:
        return

    _state.client = Posthog(
        host=settings.POSTHOG_HOST,
        project_api_key=settings.POSTHOG_KEY,
    )


def capture(
    distinct_id: str,
    event: str,
    properties: dict[str, object] | None = None,
) -> None:
    """Send one analytics event; a no-op when disabled, and never raising into the caller.

    Residency: callers must pass metadata only (model ids, counts, char lengths,
    timings) — never the embedded or reranked text.
    """
    client = _state.client
    if client is None:
        return

    try:
        client.capture(
            distinct_id=distinct_id,
            event=event,
            properties={"service": _SERVICE, **(properties or {})},
        )
    except Exception:
        logger.exception("PostHog capture failed (event=%s)", event)


def capture_exception(
    exc: BaseException,
    distinct_id: str = "server",
    properties: dict[str, object] | None = None,
) -> None:
    """Report an unhandled exception to PostHog Error Tracking; a no-op when disabled,
    and never raising into the caller.

    Residency: callers must pass metadata only (request path, method, status) — never
    the embedded or reranked text.
    """
    client = _state.client
    if client is None:
        return

    try:
        client.capture_exception(
            exc,
            distinct_id=distinct_id,
            properties={"service": _SERVICE, **(properties or {})},
        )
    except Exception:
        logger.exception("PostHog capture_exception failed")


def shutdown_analytics() -> None:
    """Flush and stop the client (called after the lifespan ``yield``)."""
    client = _state.client
    if client is None:
        return

    client.flush()
    client.shutdown()
