import logging
import re
import time
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, Request
from posthog import Posthog
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_SERVICE = "chat-bot-api"

_DISTINCT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def safe_distinct_id(raw: str | None, fallback: str) -> str:
    """The caller-supplied analytics id if it is a short opaque token, else ``fallback``.

    The header is untrusted: bounding length and charset stops a caller injecting PII,
    smuggling free text, or exploding person cardinality.
    """
    if raw is None:
        return fallback
    candidate = raw.strip()
    if _DISTINCT_ID_RE.fullmatch(candidate):
        return candidate
    return fallback


def safe_session_id(raw: str | None) -> str | None:
    """The caller-supplied PostHog session id if it is a short opaque token."""
    if raw is None:
        return None
    candidate = raw.strip()
    if _SESSION_ID_RE.fullmatch(candidate):
        return candidate
    return None


class _State:
    client: Posthog | None = None

    def __init__(self) -> None:
        self.revisions: dict[str, str] = {}


_state = _State()


def init_posthog(settings: Settings) -> None:
    _state.revisions = {
        key: value
        for key, value in {
            "app_revision": settings.APP_REVISION,
            "document_corpus_revision": settings.RAG_SYNC_EXPECTED_SOURCE_COMMIT,
        }.items()
        if re.fullmatch(r"[0-9a-f]{40}", value)
    }
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
            properties={
                **{
                    key: value
                    for key, value in (properties or {}).items()
                    if key
                    not in {
                        "app_revision",
                        "document_corpus_revision",
                        "corpus_revision",
                    }
                },
                "service": _SERVICE,
                **_state.revisions,
            },
        )
    except Exception:
        logger.exception("PostHog capture failed (event=%s)", event)


ChatPreparationPhase = Literal["credentials", "admission", "context", "agent_setup"]
ChatPreparationOutcome = Literal["denied", "error", "client_disconnect"]
ChatPreparationReason = Literal[
    "credential_required", "free_tier_unavailable", "preparation_failed", "cancelled"
]


def capture_chat_pre_stream_outcome(
    response_id: UUID,
    *,
    phase: ChatPreparationPhase,
    outcome: ChatPreparationOutcome,
    reason: ChatPreparationReason,
) -> None:
    """Request-scoped, content-free terminal preparation telemetry."""
    if (
        not isinstance(response_id, UUID)
        or phase not in {"credentials", "admission", "context", "agent_setup"}
        or (outcome, reason)
        not in {
            ("denied", "credential_required"),
            ("denied", "free_tier_unavailable"),
            ("error", "preparation_failed"),
            ("client_disconnect", "cancelled"),
        }
    ):
        return
    capture(
        str(response_id),
        "chat_pre_stream_outcome",
        {
            "response_id": str(response_id),
            "phase": phase,
            "outcome": outcome,
            "reason": reason,
            "$process_person_profile": False,
        },
    )


def capture_sponsored_event(
    distinct_id: str,
    event: str,
    *,
    response_id: str,
    mode: str,
    client_interface: str,
    outcome: str | None = None,
    admission_reason: str | None = None,
    denial_reason: str | None = None,
    provider_failure: bool | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    remaining_user_requests: int | None = None,
    remaining_global_requests: int | None = None,
) -> None:
    """Capture aggregate sponsored-chat metrics without user content or credentials."""
    properties: dict[str, object] = {
        "response_id": response_id,
        "mode": mode,
        "client_interface": client_interface,
    }
    optional_fields = {
        "outcome": outcome,
        "admission_reason": admission_reason,
        "denial_reason": denial_reason,
        "provider_failure": provider_failure,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "remaining_user_requests": remaining_user_requests,
        "remaining_global_requests": remaining_global_requests,
    }
    properties.update(
        {key: value for key, value in optional_fields.items() if value is not None},
    )
    capture(distinct_id, event, properties)


def capture_exception(
    exc: BaseException,
    distinct_id: str = "server",
    properties: dict[str, object] | None = None,
) -> None:
    # Never give the SDK the original exception, chain, message, or traceback.
    exception_type = next(
        (
            kind.__name__
            for kind in (
                TimeoutError,
                ConnectionError,
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                OSError,
            )
            if isinstance(exc, kind)
        ),
        "Exception",
    )
    capture(
        distinct_id,
        "$exception",
        {
            **(properties or {}),
            "$exception_list": [{"type": exception_type, "value": "Redacted"}],
            "$process_person_profile": False,
        },
    )


def shutdown_posthog() -> None:
    client = _state.client
    if client is None:
        return

    client.flush()
    client.shutdown()


def _request_path_template(scope: Scope) -> str:
    route = scope.get("route")
    return getattr(route, "path", None) or "unmatched"


def capture_request_exception(request: Request, exc: Exception) -> None:
    """Report an unhandled request exception (path/method metadata only, redacted body)."""
    capture_exception(
        exc,
        properties={
            "path": _request_path_template(request.scope),
            "method": request.method,
        },
    )


# Extreme-noise paths kept out of request_completed to control event volume.
_SKIP_PATHS: frozenset[str] = frozenset(
    {"/docs", "/redoc", "/openapi.json", "/favicon.ico"},
)
_SKIP_PREFIXES: tuple[str, ...] = ("/health",)


def _request_outcome(status_code: int) -> str:
    if status_code >= 500:
        return "server_error"
    if status_code >= 400:
        return "client_error"
    return "ok"


class _RequestTrackingMiddleware:
    """Emit one ``request_completed`` PostHog event per HTTP request (metadata only).

    A pure ASGI middleware, not BaseHTTPMiddleware, so it never buffers the SSE chat body:
    it only reads the status off ``http.response.start`` and times the whole request. The
    matched route TEMPLATE is reported (never the raw path), so ids never leak and route /
    person cardinality stays bounded.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in _SKIP_PATHS or path.startswith(_SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            capture(
                safe_distinct_id(Headers(scope=scope).get("x-distinct-id"), "api"),
                "request_completed",
                {
                    "route": _request_path_template(scope),
                    "method": scope.get("method", ""),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    "outcome": _request_outcome(status_code),
                },
            )


def register_request_middleware(app: FastAPI) -> None:
    """Attach the request_completed middleware (added outermost to time the whole request)."""
    app.add_middleware(_RequestTrackingMiddleware)
