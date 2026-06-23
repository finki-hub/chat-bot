import asyncio
import logging
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.llms.chat import handle_chat
from app.llms.context import get_links_context, get_retrieved_context
from app.llms.models import CHAT_MODELS
from app.schemas.chat import ChatSchema
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

_TZ = ZoneInfo(settings.TZ)
_HISTORY_TURNS_FOR_RETRIEVAL = 6
_HISTORY_TURN_MAX_CHARS = 600


def _clip(text: str) -> str:
    if len(text) <= _HISTORY_TURN_MAX_CHARS:
        return text
    return text[:_HISTORY_TURN_MAX_CHARS].rstrip() + "…"


def _history_for_retrieval(payload: ChatSchema) -> str | None:
    """Recent prior turns as a transcript, for history-aware retrieval (None if none).

    Each turn's text is clipped so a few long turns can't bloat the contextualization
    prompt (and blow a small local model's window) — reference resolution only needs the
    gist of earlier turns, not their full text.
    """
    turns = payload.history[-_HISTORY_TURNS_FOR_RETRIEVAL:]
    if not turns:
        return None
    return "\n".join(
        f"{'Корисник' if t.role == 'user' else 'Асистент'}: {_clip(t.content)}"
        for t in turns
    )


db_dep = Depends(get_db)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
    dependencies=[db_dep],
)


@router.post(
    "/",
    summary="Stream a chat response",
    description=(
        "Compute an embedding for the incoming question, retrieve top-N "
        "similar questions for context, construct a prompt, and stream back "
        "the LLM's answer as a text stream."
    ),
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "Chunked stream of SSE events",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": "data: Hello\n\ndata: World\n\n",
                    },
                },
            },
        },
    },
    operation_id="chatWithModel",
)
async def chat(
    payload: ChatSchema,
    db: Database = db_dep,
) -> StreamingResponse:
    logger.info(
        "Received chat request with payload: %s",
        payload.model_dump(mode="json", exclude_defaults=True),
    )

    history_text = _history_for_retrieval(payload)

    retrieved, links_context = await asyncio.gather(
        get_retrieved_context(
            db=db,
            query=payload.query,
            embedding_model=payload.embeddings_model,
            query_transform_model=payload.query_transform_model,
            history_text=history_text,
        ),
        get_links_context(db),
    )

    if not retrieved:
        # On a miss, leave the bare "nothing found" context alone so the prompt's
        # tool-search directive isn't diluted by the links catalog.
        context = "Не можев да пронајдам релевантни информации во базата на податоци."
    else:
        context = retrieved
        if links_context:
            context = f"{retrieved}\n\n{links_context}"

    today = datetime.now(tz=_TZ).strftime("%d.%m.%Y")
    context = f"Денешен датум: {today}.\n\n{context}"

    # Set the header on the StreamingResponse returned by handle_chat: Starlette
    # serializes headers only when the body starts streaming, so this lands first.
    # Done at the one chokepoint because the default agent path skips the SSE wrapper.
    response_id = uuid4()
    response = await handle_chat(payload, context)
    response.headers["X-Response-Id"] = str(response_id)
    return response


@router.get(
    "/models",
    summary="List available LLM models",
    description="Retrieve a list of all available LLM models for chat.",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "description": "List of available LLM models",
            "content": {
                "application/json": {
                    "example": ["llama3.3:70b", "qwen2.5:72b"],
                },
            },
        },
    },
)
def list_models() -> list[str]:
    return sorted(m.value for m in CHAT_MODELS)
