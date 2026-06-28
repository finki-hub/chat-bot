import logging
import re
import time

import torch
from fastapi import FastAPI, Request
from posthog import Posthog
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.utils.settings import Settings

logger = logging.getLogger(__name__)

_SERVICE = "chat-bot-gpu-api"

_RESPONSE_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def safe_response_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidate = raw.strip()
    return candidate if _RESPONSE_ID_RE.fullmatch(candidate) else None


def safe_distinct_id(raw: str | None) -> str | None:
    if raw is None:
        return None
    candidate = raw.strip()
    return candidate if _RESPONSE_ID_RE.fullmatch(candidate) else None


class _State:
    client: Posthog | None = None


_state = _State()


def init_analytics(settings: Settings) -> None:
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
    client = _state.client
    if client is None:
        return

    # Capture the real exception (type, message, stacktrace) for debugging.
    try:
        client.capture_exception(
            exc,
            distinct_id=distinct_id,
            properties={"service": _SERVICE, **(properties or {})},
        )
    except Exception:
        logger.exception("PostHog capture_exception failed")


def shutdown_analytics() -> None:
    client = _state.client
    if client is None:
        return

    client.flush()
    client.shutdown()


def capture_chat_inference(
    request: Request,
    *,
    stage: str,
    ms: float,
    props: dict[str, object],
) -> None:
    """Record an embed/rerank stage as an LLM-analytics span on the chat's trace.

    With a chat trace id (``X-Response-Id``) the stage is emitted as a native
    ``$ai_embedding`` / ``$ai_span`` sharing that ``$ai_trace_id`` so it nests under the
    chat's trace in PostHog; only size/latency metadata is sent, never content
    (residency). Calls without a trace id (e.g. document fills) fall back to a plain
    metadata event.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trace_id = safe_response_id(request.headers.get("X-Response-Id"))

    if trace_id is None:
        capture(
            "gpu-api",
            "chat_inference",
            {"stage": stage, "ms": ms, "device": device, **props},
        )
        return

    distinct_id = safe_distinct_id(request.headers.get("X-Distinct-Id")) or trace_id
    latency_s = round(ms / 1000.0, 4)

    if stage == "embed":
        properties: dict[str, object] = {
            "$ai_trace_id": trace_id,
            "$ai_provider": "gpu-api",
            "$ai_latency": latency_s,
            "device": device,
            **props,
        }
        model = props.get("model")
        if model is not None:
            properties["$ai_model"] = model
        capture(distinct_id, "$ai_embedding", properties)
        return

    capture(
        distinct_id,
        "$ai_span",
        {
            "$ai_trace_id": trace_id,
            "$ai_span_name": stage,
            "$ai_latency": latency_s,
            "$ai_input_state": dict(props),
            "$ai_output_state": {"reranked_documents": props.get("docs")},
            "device": device,
        },
    )


def capture_request_exception(request: Request, exc: Exception) -> None:
    """Report an unhandled request exception (path/method metadata only, redacted body)."""
    capture_exception(
        exc,
        properties={"path": request.url.path, "method": request.method},
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

    A pure ASGI middleware, not BaseHTTPMiddleware, so it never buffers a streaming body:
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
            route = scope.get("route")
            template = getattr(route, "path", None) or path
            capture(
                safe_response_id(Headers(scope=scope).get("x-response-id"))
                or "gpu-api",
                "request_completed",
                {
                    "route": template,
                    "method": scope.get("method", ""),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    "outcome": _request_outcome(status_code),
                },
            )


def register_request_middleware(app: FastAPI) -> None:
    """Attach the request_completed middleware (added outermost to time the whole request)."""
    app.add_middleware(_RequestTrackingMiddleware)
