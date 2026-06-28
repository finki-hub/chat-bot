import logging

from posthog import Posthog

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_SERVICE = "chat-bot-api"


class _State:
    client: Posthog | None = None


_state = _State()


def init_posthog(settings: Settings) -> None:
    """Construct the per-worker PostHog client.

    Call AFTER gunicorn forks (the ``post_fork`` hook): posthog-python's background
    flush thread does not survive ``os.fork()``, so a client built pre-fork silently
    drops every event in the workers. A no-op when ``POSTHOG_KEY`` is unset (dev/CI/tests).
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

    Residency: callers must pass metadata only (model ids, lengths, token counts,
    timings) — never raw prompt or answer text.
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
    """Report an unhandled exception to PostHog Error Tracking as a metadata-only
    ``$exception`` event; a no-op when disabled, and never raising into the caller.

    Residency: the exception object is NEVER handed to the SDK. posthog-python's
    ``capture_exception`` serializes the full stacktrace WITH frame-local variables and
    the raw ``str(exc)`` message, both of which can embed sovereign text (the user query,
    the answer, retrieved chunks). Only the exception type, service and caller metadata
    (request path, method, status) leave; the message/value is redacted.
    """
    client = _state.client
    if client is None:
        return

    error_type = type(exc).__name__
    try:
        client.capture(
            event="$exception",
            distinct_id=distinct_id,
            properties={
                "service": _SERVICE,
                "error_type": error_type,
                "$exception_list": [
                    {"type": error_type, "value": "(redacted for residency)"},
                ],
                **(properties or {}),
            },
        )
    except Exception:
        logger.exception("PostHog capture_exception failed")


def shutdown_posthog() -> None:
    """Flush and stop the client (the gunicorn ``worker_exit`` hook)."""
    client = _state.client
    if client is None:
        return

    client.flush()
    client.shutdown()
