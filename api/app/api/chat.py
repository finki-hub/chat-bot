import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from datetime import datetime
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.llms.agents import meta_event
from app.llms.chat import handle_chat
from app.llms.context import get_links_context, get_retrieved_context
from app.llms.models import CHAT_MODELS
from app.schemas.chat import ChatSchema
from app.utils.settings import Settings
from app.utils.timing import (
    RequestTimings,
    reset_request_timings,
    start_request_timings,
    timed,
)

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


def _sse_event_name(chunk: bytes | str | memoryview) -> str:
    """The SSE event name of ``chunk``, or '' if it has none.

    Only the first line is inspected: the rest of the frame is the (possibly large)
    data payload, and decoding bytes mid-character would risk a UnicodeDecodeError.
    """
    if isinstance(chunk, str):
        first_line = chunk.split("\n", 1)[0]
    else:
        raw = bytes(chunk)
        newline = raw.find(b"\n")
        first_line = (raw if newline == -1 else raw[:newline]).decode(errors="ignore")
    if first_line.startswith("event:"):
        return first_line[len("event:") :].strip()
    return ""


async def _instrument_stream(
    body: AsyncIterable[bytes | str | memoryview],
    *,
    payload: ChatSchema,
    response_id: UUID,
    timings: RequestTimings,
    retrieval_hit: bool,
) -> AsyncGenerator[bytes | str | memoryview]:
    """Pass the SSE body through untouched, stamping TTFT, thinking and total, then
    log one chat.timing line and emit a trailing ``meta`` frame with the same breakdown.

    Thinking time spans the first ``thinking`` frame to the first ``token`` frame, so it
    lands in the same ``meta`` diagnostics as TTFT and total.

    The ``meta`` frame trails the body's ``done`` (``total_ms`` is known only once the
    body drains); consumers that stop at ``done`` ignore it and the protocol-v2 parsers
    drop unknown events, so it stays backward compatible.
    """
    try:
        marking = True
        async for chunk in body:
            timings.mark_ttft()
            if marking:
                event_name = _sse_event_name(chunk)
                if event_name == "thinking":
                    timings.mark_thinking()
                elif event_name == "token":
                    timings.mark_answer()
                    marking = False
            yield chunk
    finally:
        timings.mark_total()
        record = timings.as_record()
        logger.info(
            "chat.timing %s",
            json.dumps(
                {
                    "response_id": str(response_id),
                    "inference_model": payload.inference_model.value,
                    "embeddings_model": payload.embeddings_model.value,
                    "query_transform_model": payload.query_transform_model.value,
                    "retrieval_hit": retrieval_hit,
                    **record,
                },
            ),
        )
    yield meta_event({"timing": record})


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

    timings, token = start_request_timings()
    try:
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
            context = (
                "Не можев да пронајдам релевантни информации во базата на податоци."
            )
        else:
            context = retrieved
            if links_context:
                context = f"{retrieved}\n\n{links_context}"

        today = datetime.now(tz=_TZ).strftime("%d.%m.%Y")
        context = f"Денешен датум: {today}.\n\n{context}"

        response_id = uuid4()
        with timed("agent.setup"):
            response = await handle_chat(payload, context)

        response.body_iterator = _instrument_stream(
            response.body_iterator,
            payload=payload,
            response_id=response_id,
            timings=timings,
            retrieval_hit=bool(retrieved),
        )

        # Set the header on the StreamingResponse returned by handle_chat: Starlette
        # serializes headers only when the body starts streaming, so this lands first.
        # Done at the one chokepoint because the default agent path skips the SSE wrapper.
        response.headers["X-Response-Id"] = str(response_id)
        return response
    finally:
        # The stream wrapper holds `timings` by closure, so the context var can be reset
        # now (before streaming) without affecting the TTFT/total it records later.
        reset_request_timings(token)


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
