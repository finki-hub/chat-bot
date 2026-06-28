import logging
import time
from asyncio import gather, to_thread
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.embeddings import router as embeddings_router
from app.api.health import router as health_router
from app.api.rerank import router as rerank_router
from app.api.streams import router as streams_router
from app.llms.bge_m3 import init_bge_m3_embedder
from app.llms.reranker import init_reranker
from app.utils.analytics import (
    capture,
    capture_exception,
    init_analytics,
    safe_response_id,
    shutdown_analytics,
)
from app.utils.exceptions import ModelNotReadyError
from app.utils.logger import setup_logging
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

setup_logging(level=settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(
            "gpu.cuda available=True device=%s count=%d",
            torch.cuda.get_device_name(0),
            torch.cuda.device_count(),
        )
    else:
        logger.warning("gpu.cuda available=False - models will run on CPU")

    init_analytics(settings)
    if not cuda_available:
        capture("gpu-api", "cuda_fallback", {"device": "cpu"})

    tasks = [to_thread(init_reranker, settings.RERANKER_MODEL)]
    if settings.PRELOAD_BGEM3:
        tasks.append(to_thread(init_bge_m3_embedder))

    await gather(*tasks)

    yield

    shutdown_analytics()


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
            {"name": "Embeddings", "description": "Manage embeddings"},
            {"name": "Health", "description": "Health check and API status"},
        ],
        host=settings.HOST,
        port=settings.PORT,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=settings.EXPOSE_HEADERS,
    )

    # Added last so it sits outermost and times the whole request. A handled error (e.g.
    # 422/503) arrives as a normal response we read the status off; an unhandled 500
    # propagates here as an exception, so status_code stays 500 and we still emit in finally.
    app.add_middleware(RequestTrackingMiddleware)

    app.include_router(embeddings_router)
    app.include_router(streams_router)
    app.include_router(rerank_router)
    app.include_router(health_router)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        raw = exc.body
        if isinstance(raw, bytes | bytearray):
            try:
                body_str = raw.decode("utf-8")
            except Exception:
                body_str = repr(raw)
        else:
            body_str = raw

        content = {"detail": exc.errors(), "body": body_str}
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder(content),
        )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_exception_handler(
        request: Request,
        exc: ModelNotReadyError,
    ) -> JSONResponse:
        logger.exception("Model not ready")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "The model is not ready. Please try again later."},
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
