import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import anyio
from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.responses import StreamingResponse
from starlette.types import Send

from app.api.discord_storage import (
    DiscordStorage,
    discord_context_metadata,
    discord_storage,
    schedule_discord_exchange,
)
from app.api.provider_credentials import (
    credential_providers_for_models,
    missing_mandatory_credential,
    resolve_provider_credentials,
)
from app.api.sponsored_access import (
    resolve_sponsored_inference,
)
from app.data.connection import Database
from app.data.db import get_db
from app.data.discord_conversations import DiscordExchange
from app.data.sponsored_usage import (
    SponsoredQuotaExceededError,
    SponsoredRequestInProgressError,
    admit_sponsored_request,
    get_sponsored_usage_snapshot,
    release_sponsored_request,
)
from app.llms.agents import (
    StreamObservation,
    error_event,
    meta_event,
    sponsored_error_event,
)
from app.llms.chat import handle_chat
from app.llms.model_access import (
    ModelAccessContext,
    SponsoredSettings,
    overlay_model_access,
)
from app.llms.model_catalog import model_catalog_service
from app.llms.model_catalog_types import ModelCatalogResponse, OllamaCatalogModel
from app.llms.models import model_id
from app.llms.ollama import fetch_ollama_catalog
from app.llms.pricing import cost_usd, is_self_hosted
from app.llms.provider_credentials import provider_for_model
from app.schemas.chat import ChatSchema
from app.schemas.sponsored_access import SponsoredQuotaSnapshot
from app.utils.async_iterators import closing_stream
from app.utils.auth import verify_api_key
from app.utils.posthog_client import (
    ChatPreparationOutcome,
    ChatPreparationPhase,
    ChatPreparationReason,
    capture,
    capture_chat_pre_stream_outcome,
    capture_sponsored_event,
    safe_distinct_id,
    safe_session_id,
)
from app.utils.settings import Settings, is_opencode_go_base_url
from app.utils.timing import (
    RequestTimings,
    record_distinct_id,
    record_response_id,
    reset_request_timings,
    start_request_timings,
    timed,
)
from app.utils.topic import classify_language, classify_script, classify_topic

logger = logging.getLogger(__name__)

settings = Settings()

_TZ = ZoneInfo(settings.TZ)
_PRE_STREAM_ERROR_MSG = (
    "Се случи грешка при подготовка на одговорот. Обидете се повторно."
)
_SPONSORED_UNAVAILABLE_MSG = "Бесплатниот пристап моментално не е достапен."
_AGENT_SEARCH_CONTEXT = "Изворите не се пребарани однапред; пребарај ги со алатката."
_SPONSORED_QUOTA_MSG = (
    "Бесплатната квота е искористена. Обидете се повторно по ресетирањето."
)
_SPONSORED_IN_PROGRESS_MSG = (
    "Веќе имате активно спонзорирано барање. Обидете се повторно за момент."
)
_CREDENTIAL_PROVIDER_LABELS = {
    "openai": "OpenAI",
    "google": "Google",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
    "openrouter": "OpenRouter",
    "opencode-go": "OpenCode Go",
}


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
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        try:
            data = json.loads(line[len(_SSE_DATA_PREFIX) :].strip())
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
_SSE_DATA_PREFIX = "data:"


def _chat_request_log_fields(
    payload: ChatSchema,
) -> dict[str, bool | float | int | str]:
    return {
        "inference_model": model_id(payload.inference_model),
        "embeddings_model": payload.embeddings_model.value,
        "reasoning": payload.reasoning,
        "interface": payload.interface,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "query_len": len(payload.query),
        "history_turns": len(payload.history),
    }


def _chat_messages(payload: ChatSchema) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.content} for turn in payload.messages]


def _chat_request_posthog_fields(payload: ChatSchema) -> dict[str, object]:
    return {
        **_chat_request_log_fields(payload),
        "message_count": len(payload.messages),
        "message_roles": [turn.role for turn in payload.messages],
    }


def _session_props(session_id: str | None) -> dict[str, object]:
    return {} if session_id is None else {"$session_id": session_id}


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
    answer_text: str,
    session_id: str | None,
    model_access_mode: Literal["sponsored", "byok", "ordinary"] = "ordinary",
) -> None:
    model = payload.inference_model
    generation_ms: float | None = None
    if timings.total_ms is not None and timings.ttft_ms is not None:
        generation_ms = max(timings.total_ms - timings.ttft_ms, 0.0)

    props: dict[str, object] = {
        "$ai_trace_id": str(response_id),
        "$ai_model": model_id(model),
        "$ai_provider": observation.provider or None,
        "$ai_input": {
            "count": len(payload.messages),
            "roles": [turn.role for turn in payload.messages],
        },
        "$ai_output_choices": [
            {"role": "assistant", "content_length": len(answer_text)},
        ],
        "provider": observation.provider or None,
        "is_self_hosted": is_self_hosted(model),
        "$ai_latency": (
            timings.total_ms / 1000.0 if timings.total_ms is not None else None
        ),
        "response_id": str(response_id),
        "retrieval_hit": retrieval_hit,
        "model_access_mode": model_access_mode,
        "outcome": outcome,
        "language": classify_language(payload.query),
        "query_script": classify_script(payload.query),
        "ttft_ms": _ms(timings.ttft_ms),
        "thinking_ms": _ms(timings.thinking_ms),
        "llm_generation_ms": _ms(generation_ms),
        "embedding_ms": _ms(timings.spans.get("retrieval.embed")),
        "retrieval_ms": _ms(timings.spans.get("retrieval.vector_search")),
        "rerank_ms": _ms(timings.spans.get("retrieval.rerank")),
        "candidate_count": timings.candidate_count,
        "top_distance": timings.top_distance,
        "transliteration_variant_added": timings.transliteration_variant_added,
        "lexical_search_outcome": timings.lexical_search_outcome,
        "dense_faq_candidate_count": timings.dense_faq_candidate_count,
        "dense_document_candidate_count": timings.dense_document_candidate_count,
        "lexical_faq_candidate_count": timings.lexical_faq_candidate_count,
        "final_faq_count": timings.final_faq_count,
        "final_document_count": timings.final_document_count,
        "lexical_only_final_count": timings.lexical_only_final_count,
        "retrieval_path": timings.retrieval_path,
        **timings.fallback_record(),
        "context_char_len": observation.context_chars,
        "context_chunk_count": len(timings.retrieval_ids),
        "answer_char_len": observation.answer_chars,
        "tool_call_count": observation.tool_call_count,
        "used_web_search": _used_web_search(observation.tool_names),
        **_session_props(session_id),
    }

    props.update(
        {
            name: score
            for name, score in {
                "reranker_min_score": settings.RERANKER_MIN_SCORE,
                "source_reranker_min_score": settings.SOURCE_RERANKER_MIN_SCORE,
            }.items()
            if 0.0 <= score <= 1.0
        }
    )

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


def _sniff_token_text(chunk: bytes | str | memoryview) -> str | None:
    if isinstance(chunk, str):
        event_name = _sse_event_name(chunk)
        text = chunk
    else:
        event_name = _sse_event_name(chunk)
        text = bytes(chunk).decode(errors="ignore")
    if event_name not in {"", "token"}:
        return None

    data_lines: list[str] = []
    for line in text.split("\n"):
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        data_lines.append(line[len(_SSE_DATA_PREFIX) :].removeprefix(" "))

    if not data_lines:
        return None
    if event_name == "":
        return "\n".join(data_lines).replace(r"\n", "\n")

    for data_line in data_lines:
        try:
            data = json.loads(data_line.strip())
        except json.JSONDecodeError:
            return None
        token_text = data.get("text") if isinstance(data, dict) else None
        if isinstance(token_text, str):
            return token_text
    return None


def _sse_frame_text(chunk: bytes | str | memoryview) -> str:
    return chunk if isinstance(chunk, str) else bytes(chunk).decode(errors="ignore")


def _complete_sse_frames(
    buffer: str,
    chunk: bytes | str | memoryview,
) -> tuple[list[str], str]:
    parts = (buffer + _sse_frame_text(chunk)).replace("\r\n", "\n").split("\n\n")
    return parts[:-1], parts[-1]


def _is_provider_error_frame(frame: str) -> bool:
    if _sse_event_name(frame) != "error":
        return False
    data = "\n".join(
        line[len(_SSE_DATA_PREFIX) :].removeprefix(" ")
        for line in frame.splitlines()
        if line.startswith(_SSE_DATA_PREFIX)
    )
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return False
    # Empty answers and credential/quota denials are not caught provider failures.
    return isinstance(payload, dict) and payload.get("code") in (
        "agent_error",
        "interrupted",
    )


def _log_sponsored_release_failure(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    # ruff: ignore[BLE001] -- task callbacks must observe and log every release failure
    except Exception as exc:
        logger.log(
            logging.ERROR,
            "Failed to release sponsored request lease error_type=%s",
            type(exc).__name__,
        )


class _ChatStreamingResponse(StreamingResponse):
    async def stream_response(self, send: Send) -> None:
        # Starlette's older-ASGI disconnect listener cancels this task. Closing
        # here (rather than in __call__) also keeps timing ContextVars in-task.
        async with closing_stream(self.body_iterator) as stream:
            self.body_iterator = stream
            await super().stream_response(send)


async def _instrument_stream(
    body: AsyncIterable[bytes | str | memoryview],
    *,
    payload: ChatSchema,
    response_id: UUID,
    timings: RequestTimings,
    distinct_id: str,
    session_id: str | None,
    observation: StreamObservation,
    sponsored_mode: str | None = None,
    model_access_mode: Literal["sponsored", "byok", "ordinary"] = "ordinary",
    storage: DiscordStorage | None = None,
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
    meta_payload: dict[str, object] = {}
    answer_parts: list[str] = []
    answer_frame_buffer = ""
    marking = True
    answered = False
    provider_failure = False
    try:
        async with closing_stream(body) as stream:
            async for chunk in stream:
                timings.mark_ttft()
                event_name = _sse_event_name(chunk)
                if marking or not answered:
                    if marking:
                        if event_name == "thinking":
                            timings.mark_thinking()
                        elif event_name == "token":
                            timings.mark_answer()
                            marking = False
                    if not answered and _is_answer_chunk(event_name, chunk):
                        answered = True
                if event_name == "reset":
                    answer_parts.clear()
                sniffed = _sniff_tokens(chunk)
                if sniffed is not None:
                    usage = sniffed
                frames, answer_frame_buffer = _complete_sse_frames(
                    answer_frame_buffer,
                    chunk,
                )
                for frame in frames:
                    if _is_provider_error_frame(frame):
                        # The agent catches provider exceptions and drains normally.
                        # A later cancellation still overrides this outcome below.
                        outcome = "provider_error"
                        provider_failure = True
                    if _sse_event_name(frame) == "reset":
                        answer_parts.clear()
                        continue
                    token_text = _sniff_token_text(frame)
                    if token_text is not None:
                        answer_parts.append(token_text)
                yield chunk
    except GeneratorExit:
        outcome = "client_disconnect"
        raise
    except asyncio.CancelledError:
        outcome = "client_disconnect"
        raise
    except Exception:
        outcome = "provider_error"
        provider_failure = True
        raise
    finally:
        timings.mark_total()
        record = timings.as_record()
        if outcome == "completed" and not answered:
            outcome = "empty_answer"
        retrieval_hit = observation.search_hit
        logger.info(
            "chat.timing %s",
            json.dumps(
                {
                    "response_id": str(response_id),
                    "inference_model": model_id(payload.inference_model),
                    "embeddings_model": payload.embeddings_model.value,
                    **timings.fallback_record(),
                    "retrieval_hit": retrieval_hit,
                    "outcome": outcome,
                    **record,
                },
            ),
        )
        effective_usage = (
            observation.usage if any(observation.usage.values()) else usage
        )
        _capture_chat_response(
            distinct_id=distinct_id,
            payload=payload,
            response_id=response_id,
            timings=timings,
            retrieval_hit=retrieval_hit,
            usage=effective_usage,
            outcome=outcome,
            observation=observation,
            answer_text="".join(answer_parts),
            session_id=session_id,
            model_access_mode=model_access_mode,
        )
        if sponsored_mode is not None:
            capture_sponsored_event(
                distinct_id,
                "sponsored_stream",
                response_id=str(response_id),
                mode=sponsored_mode,
                client_interface=payload.interface,
                outcome=outcome,
                provider_failure=provider_failure,
                input_tokens=(effective_usage or {}).get("input", 0),
                output_tokens=(effective_usage or {}).get("output", 0),
                total_tokens=(effective_usage or {}).get("total", 0),
            )
        meta_payload = {"timing": record}
        if effective_usage is not None:
            costs = cost_usd(
                payload.inference_model,
                effective_usage["input"],
                effective_usage["output"],
            )
            if costs is not None:
                meta_payload["cost"] = {
                    "input_usd": round(costs[0], 6),
                    "output_usd": round(costs[1], 6),
                    "total_usd": round(costs[2], 6),
                }
        if storage is not None:
            schedule_discord_exchange(
                storage,
                DiscordExchange(
                    conversation_id=storage.conversation_id,
                    user_id=storage.user_id,
                    response_id=response_id,
                    model=model_id(payload.inference_model),
                    question=payload.query,
                    answer="".join(answer_parts),
                    metadata={
                        "interface": "discord",
                        "responseId": str(response_id),
                        "inferenceModel": model_id(payload.inference_model),
                        "sources": [dict(source) for source in observation.sources],
                        "diagnostics": {
                            "outcome": outcome,
                            "embeddingsModel": payload.embeddings_model.value,
                            "retrievalHit": retrieval_hit,
                            "historyTurns": len(payload.history),
                            "fallbacks": timings.fallback_record(),
                            "retrieval": {
                                **timings.retrieval_record(),
                                "reranker_min_score": settings.RERANKER_MIN_SCORE,
                                "source_reranker_min_score": settings.SOURCE_RERANKER_MIN_SCORE,
                            },
                            "timing": record,
                            "usage": effective_usage,
                            "cost": meta_payload.get("cost"),
                            "toolNames": sorted(observation.tool_names),
                            "toolCallCount": observation.tool_call_count,
                            "contextChars": observation.context_chars,
                            "provider": observation.provider or None,
                            "modelAccessMode": model_access_mode,
                            "finishReason": observation.finish_reason or None,
                        },
                        "discord": discord_context_metadata(payload),
                    },
                    history_generation=storage.history_generation,
                ),
            )
    yield meta_event(meta_payload)


db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@dataclass(slots=True)
class _PreparationOutcome:
    response_id: UUID
    phase: ChatPreparationPhase = "credentials"
    handled: bool = False

    def capture(
        self,
        outcome: ChatPreparationOutcome,
        reason: ChatPreparationReason,
    ) -> None:
        if not self.handled:
            self.handled = True
            capture_chat_pre_stream_outcome(
                self.response_id,
                phase=self.phase,
                outcome=outcome,
                reason=reason,
            )


async def _chat_response_stream(
    payload: ChatSchema,
    request: Request,
    db: Database,
    response_id: UUID,
) -> AsyncGenerator[bytes | str | memoryview]:
    preparation = _PreparationOutcome(response_id)
    try:
        async with closing_stream(
            _prepare_chat_response_stream(
                payload, request, db, response_id, preparation
            ),
        ) as stream:
            async for chunk in stream:
                yield chunk
    except asyncio.CancelledError, GeneratorExit:
        preparation.capture("client_disconnect", "cancelled")
        raise
    except Exception:
        preparation.capture("error", "preparation_failed")
        raise


async def _prepare_chat_response_stream(
    payload: ChatSchema,
    request: Request,
    db: Database,
    response_id: UUID,
    preparation: _PreparationOutcome,
) -> AsyncGenerator[bytes | str | memoryview]:
    """Build context (streaming retrieval status), then yield the agent answer stream."""
    logger.info(
        "Received chat request: %s",
        json.dumps(_chat_request_log_fields(payload), sort_keys=True),
    )
    settings: Settings = request.app.state.settings
    sponsored_model_id = settings.SPONSORED_MODEL_ID

    supplied_conversation_id = request.headers.get("x-conversation-id")
    try:
        conversation_id = (
            str(UUID(supplied_conversation_id))
            if (supplied_conversation_id is not None)
            else None
        )
    except ValueError:
        preparation.capture("denied", "invalid_conversation_id")
        yield error_event("invalid_conversation_id", "InvalidConversationID")
        return

    storage = await discord_storage(
        payload,
        db,
        conversation_id,
        getattr(request.app.state, "discord_writes", None),
    )
    if conversation_id is None and "opencode-go" in credential_providers_for_models(
        payload.inference_model,
    ):
        preparation.capture("denied", "missing_conversation_id")
        yield error_event("missing_conversation_id", "MissingConversationID")
        return
    credentials = await resolve_provider_credentials(
        db,
        user_id=payload.user_id,
        providers=credential_providers_for_models(
            payload.embeddings_model,
            payload.inference_model,
        ),
        settings=settings,
    )
    if credentials is not None and conversation_id is not None:
        credentials = replace(credentials, conversation_id=conversation_id)
    inference_provider = provider_for_model(payload.inference_model)
    user_inference_credential = (
        None
        if credentials is None or inference_provider is None
        else credentials.for_provider(inference_provider)
    )
    user_credential_rejected = (
        inference_provider is not None
        and credentials is not None
        and inference_provider in credentials.rejected_providers
    )
    inference_resolution = resolve_sponsored_inference(
        payload.inference_model,
        user_inference_credential,
        settings,
        user_credential_rejected=user_credential_rejected,
    )
    if (
        inference_resolution.sponsored
        and is_opencode_go_base_url(settings.SPONSORED_MODEL_BASE_URL)
        and conversation_id is None
    ):
        preparation.capture("denied", "missing_conversation_id")
        yield error_event("missing_conversation_id", "MissingConversationID")
        return

    missing_credential = missing_mandatory_credential(
        credentials,
        inference_model=payload.inference_model,
        embeddings_model=payload.embeddings_model,
        inference_credential=inference_resolution.credential,
    )
    if missing_credential is not None:
        if (
            model_id(payload.inference_model) == sponsored_model_id
            and payload.user_id is not None
            and missing_credential.stage == "inference"
            and not user_credential_rejected
        ):
            preparation.capture("denied", "free_tier_unavailable")
            yield sponsored_error_event(
                "free_tier_unavailable",
                _SPONSORED_UNAVAILABLE_MSG,
            )
            return
        provider_label = _CREDENTIAL_PROVIDER_LABELS[missing_credential.provider]
        preparation.capture("denied", "credential_required")
        yield error_event(
            "credential_required",
            f"Потребен е ваш {provider_label} API клуч. Додајте го во поставките за провајдери.",
            provider=missing_credential.provider,
            stage=missing_credential.stage,
        )
        return

    timings, token = start_request_timings()
    distinct_id = safe_distinct_id(
        request.headers.get("X-Distinct-Id"),
        str(response_id),
    )
    sponsored_mode = (
        "byok" if model_id(payload.inference_model) == sponsored_model_id else None
    )
    admitted_sponsored_user_id: UUID | None = None
    accepted = False
    storage_handed_off = False
    early_outcome = "provider_error"
    try:
        preparation.phase = "admission"
        if inference_resolution.sponsored:
            if payload.user_id is None:
                preparation.capture("denied", "credential_required")
                yield error_event(
                    "credential_required",
                    f"Потребен е ваш {_CREDENTIAL_PROVIDER_LABELS[settings.SPONSORED_MODEL_PROVIDER]} API клуч. Додајте го во поставките за провајдери.",
                    provider=settings.SPONSORED_MODEL_PROVIDER,
                    stage="inference",
                )
                return
            global_limit = settings.SPONSORED_DAILY_GLOBAL_LIMIT
            if global_limit is None:
                preparation.capture("denied", "free_tier_unavailable")
                yield sponsored_error_event(
                    "free_tier_unavailable",
                    _SPONSORED_UNAVAILABLE_MSG,
                )
                return
            try:
                admission = await admit_sponsored_request(
                    db,
                    user_id=payload.user_id,
                    request_id=response_id,
                    user_limit=settings.SPONSORED_DAILY_USER_LIMIT,
                    global_limit=global_limit,
                    lease_ttl=timedelta(
                        seconds=settings.SPONSORED_REQUEST_LEASE_SECONDS,
                    ),
                )
            except SponsoredRequestInProgressError:
                # Existing sponsored_denied owns this outcome.
                preparation.handled = True
                capture_sponsored_event(
                    distinct_id,
                    "sponsored_denied",
                    response_id=str(response_id),
                    mode="sponsored",
                    client_interface=payload.interface,
                    denial_reason="sponsored_request_in_progress",
                    outcome="denied",
                )
                yield sponsored_error_event(
                    "sponsored_request_in_progress",
                    _SPONSORED_IN_PROGRESS_MSG,
                )
                return
            except SponsoredQuotaExceededError as exc:
                preparation.handled = True
                capture_sponsored_event(
                    distinct_id,
                    "sponsored_denied",
                    response_id=str(response_id),
                    mode="sponsored",
                    client_interface=payload.interface,
                    denial_reason="free_quota_exhausted",
                    outcome="denied",
                )
                yield sponsored_error_event(
                    "free_quota_exhausted",
                    _SPONSORED_QUOTA_MSG,
                    resets_at=exc.reset_at,
                )
                return
            admitted_sponsored_user_id = payload.user_id
            sponsored_mode = "sponsored"
            capture_sponsored_event(
                distinct_id,
                "sponsored_admitted",
                response_id=str(response_id),
                mode="sponsored",
                client_interface=payload.interface,
                admission_reason="quota_available",
                outcome="admitted",
                remaining_user_requests=admission.snapshot.remaining_user_requests,
                remaining_global_requests=admission.snapshot.remaining_global_requests,
            )

        accepted = True
        preparation.phase = "context"
        session_id = safe_session_id(request.headers.get("X-PostHog-Session-Id"))
        record_distinct_id(distinct_id)
        observation = StreamObservation(
            distinct_id=distinct_id,
            response_id=str(response_id),
        )
        capture(
            distinct_id,
            "chat_request",
            {
                "response_id": str(response_id),
                **_chat_request_posthog_fields(payload),
                **_session_props(session_id),
            },
        )
        capture(
            distinct_id,
            "query_classified",
            {
                "response_id": str(response_id),
                "topic": classify_topic(payload.query),
                **_session_props(session_id),
            },
        )

        try:
            today = datetime.now(tz=_TZ).strftime("%d.%m.%Y")
            context = f"Денешен датум: {today}.\n\n{_AGENT_SEARCH_CONTEXT}"

            preparation.phase = "agent_setup"
            with timed("agent.setup"):
                response = await handle_chat(
                    payload,
                    context,
                    observation=observation,
                    db=db,
                    inference_credential=inference_resolution.credential,
                    upstream_model=inference_resolution.upstream_model,
                    **(
                        {"conversation_id": conversation_id}
                        if conversation_id is not None
                        else {}
                    ),
                    max_tokens=(
                        min(payload.max_tokens, inference_resolution.max_output_tokens)
                        if inference_resolution.max_output_tokens is not None
                        else None
                    ),
                    retrieval_credentials=credentials,
                )
        # ruff: ignore[BLE001] -- the stream boundary converts every provider failure to a safe event
        except Exception as exc:
            preparation.capture("error", "preparation_failed")
            logger.log(
                logging.ERROR,
                "Chat agent setup failed before streaming error_type=%s",
                type(exc).__name__,
            )
            yield error_event("agent_error", _PRE_STREAM_ERROR_MSG)
            return

        model_access_mode: Literal["sponsored", "byok", "ordinary"]
        if inference_resolution.sponsored:
            model_access_mode = "sponsored"
        elif user_inference_credential is not None:
            model_access_mode = "byok"
        else:
            model_access_mode = "ordinary"

        response.body_iterator = _instrument_stream(
            response.body_iterator,
            payload=payload,
            response_id=response_id,
            timings=timings,
            distinct_id=distinct_id,
            observation=observation,
            session_id=session_id,
            sponsored_mode=sponsored_mode,
            model_access_mode=model_access_mode,
            storage=storage,
        )

        # No suspension between handoff and first iteration: generation owns all
        # subsequent terminal outcomes, including provider failure/disconnect.
        async with closing_stream(response.body_iterator) as stream:
            preparation.handled = True
            storage_handed_off = True
            async for chunk in stream:
                yield chunk
    except asyncio.CancelledError, GeneratorExit:
        early_outcome = "client_disconnect"
        raise
    finally:
        if accepted and storage is not None and not storage_handed_off:
            timings.mark_total()
            schedule_discord_exchange(
                storage,
                DiscordExchange(
                    conversation_id=storage.conversation_id,
                    user_id=storage.user_id,
                    response_id=response_id,
                    model=model_id(payload.inference_model),
                    question=payload.query,
                    answer="",
                    history_generation=storage.history_generation,
                    metadata={
                        "interface": "discord",
                        "responseId": str(response_id),
                        "inferenceModel": model_id(payload.inference_model),
                        "sources": [],
                        "discord": discord_context_metadata(payload),
                        "diagnostics": {
                            "outcome": early_outcome,
                            "phase": preparation.phase,
                            "embeddingsModel": payload.embeddings_model.value,
                            "historyTurns": len(payload.history),
                            "timing": timings.as_record(),
                            "retrieval": {
                                **timings.retrieval_record(),
                                "reranker_min_score": settings.RERANKER_MIN_SCORE,
                                "source_reranker_min_score": settings.SOURCE_RERANKER_MIN_SCORE,
                            },
                            "fallbacks": timings.fallback_record(),
                            "usage": None,
                            "cost": None,
                        },
                    },
                ),
            )
        if admitted_sponsored_user_id is not None:
            release_task = asyncio.create_task(
                release_sponsored_request(
                    db,
                    user_id=admitted_sponsored_user_id,
                    request_id=response_id,
                ),
            )
            try:
                with anyio.CancelScope(shield=True):
                    await asyncio.shield(release_task)
            except asyncio.CancelledError:
                release_task.add_done_callback(_log_sponsored_release_failure)
                raise
            # ruff: ignore[BLE001] -- lease cleanup must log every backend failure without masking cancellation
            except Exception as exc:
                logger.log(
                    logging.ERROR,
                    "Failed to release sponsored request lease error_type=%s",
                    type(exc).__name__,
                )
        reset_request_timings(token)


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
    dependencies=[api_key_dep],
)
async def chat(
    payload: ChatSchema,
    request: Request,
    x_response_id: Annotated[
        UUID | None,
        Header(alias="X-Response-Id"),
    ] = None,
    db: Database = db_dep,
) -> StreamingResponse:
    response_id = x_response_id or uuid4()
    record_response_id(str(response_id))
    stream = _chat_response_stream(payload, request, db, response_id)
    response = _ChatStreamingResponse(stream, media_type="text/event-stream")
    response.headers["X-Response-Id"] = str(response_id)
    return response


@router.get(
    "/models",
    summary="List available LLM models",
    description="Retrieve a list of all available LLM models for chat.",
    status_code=status.HTTP_200_OK,
    dependencies=[api_key_dep],
)
async def list_models(
    request: Request,
    user_id: UUID | None = None,
    db: Database = db_dep,
) -> ModelCatalogResponse:
    settings: Settings = request.app.state.settings
    credentials = await resolve_provider_credentials(
        db,
        user_id=user_id,
        providers=frozenset(
            {"openai", "google", "anthropic", "ollama", "openrouter", "opencode-go"},
        ),
        settings=settings,
    )
    ollama_models: tuple[OllamaCatalogModel, ...] = ()
    if credentials is not None and credentials.ollama is not None:
        ollama_models = await fetch_ollama_catalog(credentials.ollama)
    usage_snapshot = None
    global_limit = settings.SPONSORED_DAILY_GLOBAL_LIMIT
    if (
        user_id is not None
        and settings.SPONSORED_MODEL_ENABLED
        and global_limit is not None
    ):
        usage_snapshot = await get_sponsored_usage_snapshot(
            db,
            user_id=user_id,
            user_limit=settings.SPONSORED_DAILY_USER_LIMIT,
            global_limit=global_limit,
        )

    personal_quota = None
    global_quota = None
    utc_reset = datetime.now(UTC)
    if usage_snapshot is not None:
        personal_quota = SponsoredQuotaSnapshot(
            limit=usage_snapshot.user_limit,
            remaining=usage_snapshot.remaining_user_requests,
            resets_at=usage_snapshot.reset_at,
        )
        global_quota = SponsoredQuotaSnapshot(
            limit=usage_snapshot.global_limit,
            remaining=usage_snapshot.remaining_global_requests,
            resets_at=usage_snapshot.reset_at,
        )
        utc_reset = usage_snapshot.reset_at

    access_context = ModelAccessContext(
        available_providers=(
            credentials.available_providers()
            if credentials is not None
            else frozenset()
        ),
        rejected_providers=(
            credentials.rejected_providers if credentials is not None else frozenset()
        ),
        sponsored_settings=SponsoredSettings(
            enabled=settings.SPONSORED_MODEL_ENABLED,
            provider_configured=(
                settings.SPONSORED_MODEL_API_KEY is not None
                and bool(settings.SPONSORED_MODEL_API_KEY.get_secret_value().strip())
            ),
        ),
        sponsored_model_id=settings.SPONSORED_MODEL_ID,
        sponsored_provider=settings.SPONSORED_MODEL_PROVIDER,
        personal_quota=personal_quota,
        global_quota=global_quota,
        utc_reset=utc_reset,
    )
    catalog = await model_catalog_service.get_catalog(ollama_models)
    return catalog.model_copy(
        update={
            "models": tuple(
                overlay_model_access(model, access_context) for model in catalog.models
            ),
        },
    )
