import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from datetime import datetime
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.db import get_db
from app.llms.agents import StreamObservation, meta_event
from app.llms.chat import handle_chat
from app.llms.context import get_links_context, get_retrieved_context
from app.llms.models import CHAT_MODELS
from app.llms.pricing import cost_usd, is_self_hosted
from app.schemas.chat import ChatSchema
from app.utils.posthog_client import capture, safe_distinct_id
from app.utils.settings import Settings
from app.utils.timing import (
    RequestTimings,
    record_distinct_id,
    record_response_id,
    reset_request_timings,
    start_request_timings,
    timed,
)
from app.utils.topic import classify_language, classify_topic

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


def _is_answer_chunk(event_name: str, chunk: bytes | str | memoryview) -> bool:
    if event_name == "token":
        return True
    return event_name == "" and bool(chunk)


_META_PREFIX = "event: meta"
_META_PREFIX_BYTES = _META_PREFIX.encode()


def _sniff_tokens(chunk: bytes | str | memoryview) -> dict[str, int] | None:
    if isinstance(chunk, str):
        if not chunk.startswith(_META_PREFIX):
            return None
        text = chunk
    else:
        if bytes(chunk[: len(_META_PREFIX_BYTES)]) != _META_PREFIX_BYTES:
            return None
        text = bytes(chunk).decode(errors="ignore")
    for line in text.split("\n"):
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[len("data:") :].strip())
        except json.JSONDecodeError:
            return None
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if isinstance(tokens, dict):
            return {
                "input": int(tokens.get("input", 0) or 0),
                "output": int(tokens.get("output", 0) or 0),
            }
    return None


def _ms(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


_WEB_SEARCH_HINTS = ("web_search", "web-search", "websearch", "search_web", "tavily")


def _used_web_search(tool_names: set[str]) -> bool:
    return any(
        hint in name.lower() for name in tool_names for hint in _WEB_SEARCH_HINTS
    )


def _capture_chat_response(
    *,
    distinct_id: str,
    payload: ChatSchema,
    response_id: UUID,
    timings: RequestTimings,
    retrieval_hit: bool,
    usage: dict[str, int] | None,
    outcome: str,
    observation: StreamObservation,
) -> None:
    model = payload.inference_model
    generation_ms: float | None = None
    if timings.total_ms is not None and timings.ttft_ms is not None:
        generation_ms = max(timings.total_ms - timings.ttft_ms, 0.0)

    props: dict[str, object] = {
        "$ai_trace_id": str(response_id),
        "$ai_model": model.value,
        "$ai_provider": observation.provider or None,
        "provider": observation.provider or None,
        "is_self_hosted": is_self_hosted(model),
        "$ai_latency": (
            timings.total_ms / 1000.0 if timings.total_ms is not None else None
        ),
        "response_id": str(response_id),
        "retrieval_hit": retrieval_hit,
        "outcome": outcome,
        "language": classify_language(payload.query),
        "ttft_ms": _ms(timings.ttft_ms),
        "thinking_ms": _ms(timings.thinking_ms),
        "llm_generation_ms": _ms(generation_ms),
        "embedding_ms": _ms(timings.spans.get("retrieval.embed")),
        "retrieval_ms": _ms(timings.spans.get("retrieval.vector_search")),
        "rerank_ms": _ms(timings.spans.get("retrieval.rerank")),
        "query_transform_ms": _ms(timings.spans.get("retrieval.rewrite_hyde")),
        "candidate_count": timings.candidate_count,
        "top_distance": timings.top_distance,
        "context_char_len": observation.context_chars,
        "context_chunk_count": len(timings.retrieval_ids),
        "answer_char_len": observation.answer_chars,
        "tool_call_count": observation.tool_call_count,
        "used_web_search": _used_web_search(observation.tool_names),
    }

    if observation.finish_reason:
        props["finish_reason"] = observation.finish_reason
        props["truncated"] = observation.finish_reason == "length"

    if timings.reranker_score_max is not None:
        props["reranker_score_max"] = round(timings.reranker_score_max, 4)
        props["reranker_score_min"] = round(timings.reranker_score_min or 0.0, 4)
        props["chunks_above_threshold"] = timings.reranker_above_threshold

    if usage is not None:
        input_tokens = usage["input"]
        output_tokens = usage["output"]
        props["$ai_input_tokens"] = input_tokens
        props["$ai_output_tokens"] = output_tokens
        if usage.get("cache_read"):
            props["$ai_cache_read_input_tokens"] = usage["cache_read"]
        if usage.get("reasoning"):
            props["$ai_reasoning_tokens"] = usage["reasoning"]
        if generation_ms and output_tokens:
            props["output_tokens_per_sec"] = round(
                output_tokens / (generation_ms / 1000.0),
                1,
            )
        costs = cost_usd(model, input_tokens, output_tokens)
        if costs is not None:
            props["$ai_input_cost_usd"] = round(costs[0], 6)
            props["$ai_output_cost_usd"] = round(costs[1], 6)
            props["$ai_total_cost_usd"] = round(costs[2], 6)
            props["cost_known"] = True
        else:
            props["cost_known"] = False
    else:
        props["cost_known"] = False

    capture(distinct_id, "$ai_generation", props)


async def _instrument_stream(
    body: AsyncIterable[bytes | str | memoryview],
    *,
    payload: ChatSchema,
    response_id: UUID,
    timings: RequestTimings,
    retrieval_hit: bool,
    distinct_id: str,
    observation: StreamObservation,
) -> AsyncGenerator[bytes | str | memoryview]:
    """Pass the SSE body through untouched, stamping TTFT, thinking and total, then
    log one chat.timing line and emit a trailing ``meta`` frame with the same breakdown.

    Thinking time spans the first ``thinking`` frame to the first ``token`` frame, so it
    lands in the same ``meta`` diagnostics as TTFT and total.

    The ``meta`` frame trails the body's ``done`` (``total_ms`` is known only once the
    body drains); consumers that stop at ``done`` ignore it and the protocol-v2 parsers
    drop unknown events, so it stays backward compatible.
    """
    outcome = "completed"
    usage: dict[str, int] | None = None
    marking = True
    answered = False
    try:
        async for chunk in body:
            timings.mark_ttft()
            if marking or not answered:
                event_name = _sse_event_name(chunk)
                if marking:
                    if event_name == "thinking":
                        timings.mark_thinking()
                    elif event_name == "token":
                        timings.mark_answer()
                        marking = False
                if not answered and _is_answer_chunk(event_name, chunk):
                    answered = True
            sniffed = _sniff_tokens(chunk)
            if sniffed is not None:
                usage = sniffed
            yield chunk
    except GeneratorExit:
        outcome = "client_disconnect"
        raise
    except asyncio.CancelledError:
        outcome = "client_disconnect"
        raise
    finally:
        timings.mark_total()
        record = timings.as_record()
        if outcome == "completed" and not answered:
            outcome = "empty_answer"
        logger.info(
            "chat.timing %s",
            json.dumps(
                {
                    "response_id": str(response_id),
                    "inference_model": payload.inference_model.value,
                    "embeddings_model": payload.embeddings_model.value,
                    "query_transform_model": payload.query_transform_model.value,
                    "retrieval_hit": retrieval_hit,
                    "outcome": outcome,
                    **record,
                },
            ),
        )
        _capture_chat_response(
            distinct_id=distinct_id,
            payload=payload,
            response_id=response_id,
            timings=timings,
            retrieval_hit=retrieval_hit,
            usage=observation.usage if any(observation.usage.values()) else usage,
            outcome=outcome,
            observation=observation,
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
    request: Request,
    db: Database = db_dep,
) -> StreamingResponse:
    logger.info(
        "Received chat request with payload: %s",
        payload.model_dump(mode="json", exclude_defaults=True),
    )

    history_text = _history_for_retrieval(payload)

    timings, token = start_request_timings()
    try:
        response_id = uuid4()
        record_response_id(str(response_id))
        distinct_id = safe_distinct_id(
            request.headers.get("X-Distinct-Id"),
            str(response_id),
        )
        record_distinct_id(distinct_id)
        observation = StreamObservation(
            distinct_id=distinct_id,
            response_id=str(response_id),
        )
        capture(
            distinct_id,
            "chat_request",
            {
                "inference_model": payload.inference_model.value,
                "embeddings_model": payload.embeddings_model.value,
                "query_transform_model": payload.query_transform_model.value,
                "reasoning": payload.reasoning,
                "temperature": payload.temperature,
                "max_tokens": payload.max_tokens,
                "query_len": len(payload.query),
                "history_turns": len(payload.history),
            },
        )
        capture(
            distinct_id,
            "query_classified",
            {
                "response_id": str(response_id),
                "topic": classify_topic(payload.query),
            },
        )

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
            capture(
                distinct_id,
                "retrieval_miss",
                {
                    "response_id": str(response_id),
                    "embeddings_model": payload.embeddings_model.value,
                    "query_transform_model": payload.query_transform_model.value,
                    "candidate_count": timings.candidate_count,
                    "top_distance": timings.top_distance,
                    "query_len": len(payload.query),
                },
            )
            # On a miss, leave the bare "nothing found" context alone so the prompt's
            # tool-search directive isn't diluted by the links catalog.
            context = (
                "Не можев да пронајдам релевантни информации во базата на податоци."
            )
        else:
            context = retrieved
            observation.context_chars = len(retrieved)
            if links_context:
                context = f"{retrieved}\n\n{links_context}"
            if timings.retrieval_ids:
                capture(
                    distinct_id,
                    "retrieval_used",
                    {
                        "response_id": str(response_id),
                        "embeddings_model": payload.embeddings_model.value,
                        "retrieval_ids": timings.retrieval_ids,
                        "retrieval_count": len(timings.retrieval_ids),
                    },
                )

        today = datetime.now(tz=_TZ).strftime("%d.%m.%Y")
        context = f"Денешен датум: {today}.\n\n{context}"

        with timed("agent.setup"):
            response = await handle_chat(payload, context, observation)

        response.body_iterator = _instrument_stream(
            response.body_iterator,
            payload=payload,
            response_id=response_id,
            timings=timings,
            retrieval_hit=bool(retrieved),
            distinct_id=distinct_id,
            observation=observation,
        )

        # Set the header on the StreamingResponse returned by handle_chat: Starlette
        # serializes headers only when the body starts streaming, so this lands first.
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
