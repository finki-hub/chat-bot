import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.chat import router as chat_router
from app.api.diplomas import router as diplomas_router
from app.api.documents import router as documents_router
from app.api.feedback import router as feedback_router
from app.api.groups import router as groups_router
from app.api.health import router as health_router
from app.api.links import router as links_router
from app.api.questions import router as questions_router
from app.api.recommendations import router as recommendations_router
from app.data.connection import Database
from app.utils.exceptions import RetrievalError
from app.utils.http_client import close_http_client, init_http_client
from app.utils.logger import setup_logging
from app.utils.posthog_client import capture, capture_exception, safe_distinct_id
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

setup_logging(level=settings.LOG_LEVEL)

# Auth secrets with guessable built-in defaults. Warn (not fail) at startup so dev still
# boots, but a misconfigured prod doesn't silently ship a known key.
_INSECURE_SECRET_DEFAULTS: dict[str, str] = {
    "API_KEY": "your_api_key_here",
    "MCP_API_KEY": "SystemPass",
}


def _warn_on_insecure_defaults(current: Settings) -> None:
    for name, default in _INSECURE_SECRET_DEFAULTS.items():
        if getattr(current, name) == default:
            logger.warning(
                "%s is using its insecure built-in default; set it via the environment "
                "to protect authenticated endpoints.",
                name,
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    App startup/shutdown: init DB, shared HTTP client, and run migrations.
    """
    _warn_on_insecure_defaults(settings)

    db = Database(dsn=settings.DATABASE_URL)
    app.state.db = db

    await db.init()
    init_http_client()

    yield

    await db.disconnect()
    await close_http_client()


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


class RequestTrackingMiddleware:
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
            # FastAPI puts the matched APIRoute (template path) on the scope; absent on a
            # 404, where the raw path is the only thing available.
            route = scope.get("route")
            template = getattr(route, "path", None) or path
            capture(
                safe_distinct_id(Headers(scope=scope).get("x-distinct-id"), "api"),
                "request_completed",
                {
                    "route": template,
                    "method": scope.get("method", ""),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    "outcome": _request_outcome(status_code),
                },
            )


def make_app(settings: Settings) -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.API_VERSION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Chat", "description": "Chat with LLMs"},
            {"name": "Feedback", "description": "Submit response feedback"},
            {"name": "Questions", "description": "Manage questions"},
            {"name": "Documents", "description": "Manage source-of-truth documents"},
            {"name": "Diplomas", "description": "Manage diplomas"},
            {
                "name": "Recommendations",
                "description": "Recommend thesis committees",
            },
            {
                "name": "Groups",
                "description": "Temporal staff groups (who works together, by era)",
            },
            {"name": "Links", "description": "Manage links"},
            {"name": "Health", "description": "Health check and API status"},
        ],
        host=settings.HOST,
        port=settings.PORT,
    )
    app.state.settings = settings

    # Wildcard origin + credentials is browser-rejected and unsafe; enable credentials
    # only for an explicit allowlist.
    allow_credentials = "*" not in settings.ALLOWED_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=settings.EXPOSE_HEADERS,
    )

    # Added last so it sits outermost and times the whole request. A handled error (e.g.
    # 422/503) arrives as a normal response we read the status off; an unhandled 500
    # propagates here as an exception, so status_code stays 500 and we still emit in finally.
    app.add_middleware(RequestTrackingMiddleware)

    app.include_router(health_router)
    app.include_router(questions_router)
    app.include_router(documents_router)
    app.include_router(diplomas_router)
    app.include_router(recommendations_router)
    app.include_router(groups_router)
    app.include_router(links_router)
    app.include_router(chat_router)
    app.include_router(feedback_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        # Don't echo the raw request body back (avoids reflecting arbitrary/oversized input).
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder({"detail": exc.errors()}),
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_exception_handler(
        request: Request,
        exc: RetrievalError,
    ) -> JSONResponse:
        logger.exception("Context retrieval failed")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Failed to retrieve or re-rank context for the query."},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("Unhandled exception")
        capture_exception(
            exc,
            properties={"path": request.url.path, "method": request.method},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."},
        )

    return app


app = make_app(settings)
