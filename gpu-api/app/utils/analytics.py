import logging
import re

from posthog import Posthog

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_SERVICE = "chat-bot-gpu-api"

# X-Response-Id is an untrusted client header used as the PostHog person id. Bound it to a
# short, opaque token (the api forwards a UUID) so a caller can't inject PII/free text or
# explode person cardinality.
_RESPONSE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def safe_response_id(raw: str | None) -> str | None:
    """The caller's ``X-Response-Id`` if it's a short opaque token, else ``None``."""
    if raw is None:
        return None
    candidate = raw.strip()
    return candidate if _RESPONSE_ID_RE.fullmatch(candidate) else None


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
    """Report an unhandled exception to PostHog Error Tracking as a metadata-only
    ``$exception`` event; a no-op when disabled, and never raising into the caller.

    Residency: the exception object is NEVER handed to the SDK. posthog-python's
    ``capture_exception`` serializes the full stacktrace WITH frame-local variables and
    the raw ``str(exc)`` message, both of which can embed sovereign text (the embedded or
    reranked query/documents). Only the exception type, service and caller metadata
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


def shutdown_analytics() -> None:
    """Flush and stop the client (called after the lifespan ``yield``)."""
    client = _state.client
    if client is None:
        return

    client.flush()
    client.shutdown()
