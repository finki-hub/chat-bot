import logging

from posthog import Posthog

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_SERVICE = "chat-bot-api"


class _State:
    client: Posthog | None = None


_state = _State()


def init_posthog(settings: Settings) -> None:
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
    # Residency: never forward the exc object — posthog-python serialises frame-locals + str(exc), which leak sovereign query/answer/retrieved text. Send a redacted $exception instead.
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
    client = _state.client
    if client is None:
        return

    client.flush()
    client.shutdown()
