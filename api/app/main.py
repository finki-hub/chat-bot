import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.chat_state import router as chat_state_router
from app.api.chat_title import router as chat_title_router
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
from app.utils.posthog_client import (
    capture_request_exception,
    register_request_middleware,
)
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

setup_logging(level=settings.LOG_LEVEL)


def _warn_on_insecure_defaults(current: Settings) -> None:
    insecure_names = current.insecure_secret_names()
    if not insecure_names:
        return

    logger.warning(
        "One or more authentication secrets are using insecure built-in defaults; "
        "set non-default values to protect authenticated endpoints.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """
    App startup/shutdown: init DB, shared HTTP client, and run migrations.
    """
    current_settings: Settings = app.state.settings
    _warn_on_insecure_defaults(current_settings)

    db = Database(dsn=current_settings.DATABASE_URL)
    app.state.db = db

    await db.init()
    init_http_client()

    yield

    await db.disconnect()
    await close_http_client()


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

    # Added last so it sits outermost and times the whole request.
    register_request_middleware(app)

    app.include_router(health_router)
    app.include_router(questions_router)
    app.include_router(documents_router)
    app.include_router(diplomas_router)
    app.include_router(recommendations_router)
    app.include_router(groups_router)
    app.include_router(links_router)
    app.include_router(chat_router)
    app.include_router(chat_state_router)
    app.include_router(chat_title_router)
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
        capture_request_exception(request, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An unexpected internal server error occurred."},
        )

    return app


app = make_app(settings)
