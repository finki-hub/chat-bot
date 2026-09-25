import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import anyio
import pytest
from fastapi import Request

from app.api import chat as chat_api
from app.data.connection import Database
from app.llms import context, query_variants
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.query_modes import QueryTransformMode
from app.schemas.chat import ChatSchema
from app.schemas.chat_credentials import ChatCredentialSecret
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.timing import reset_request_timings, start_request_timings
from tests.chat_models_access_support import settings


@pytest.mark.parametrize(
    ("mode", "credential_provider", "outputs", "reason", "effective"),
    [
        ("raw", None, (), "not_requested", "raw"),
        ("raw", "openai", (), "not_requested", "raw"),
        ("rewrite_hyde", None, (), "credential_missing", "raw"),
        ("rewrite_hyde", "google", (), "credential_missing", "raw"),
        ("rewrite_hyde", "openai", ("query", " "), "no_usable_variants", "raw"),
        ("rewrite_hyde", "openai", (None, None), "no_usable_variants", "raw"),
        ("rewrite_hyde", "openai", ("rewrite", None), "partial_variants", "rewrite"),
        ("rewrite_hyde", "openai", (None, "passage"), "partial_variants", "hyde"),
        ("rewrite_hyde", "openai", ("rewrite", "passage"), "none", "rewrite_hyde"),
        ("rewrite", "openai", ("rewrite", None), "none", "rewrite"),
        ("hyde", "openai", (None, "passage"), "none", "hyde"),
    ],
)
@pytest.mark.parametrize("search_fails", [False, True])
def test_transform_diagnostics_survive_no_candidates_and_search_errors(
    *,
    monkeypatch,
    caplog,
    mode,
    credential_provider,
    outputs,
    reason,
    effective,
    search_fails,
):
    calls = []
    credential = (
        ChatCredentialSecret(provider=credential_provider, api_key="private-test-key")
        if credential_provider
        else None
    )
    credentials = LlmProviderCredentials(
        openai=credential if credential_provider == "openai" else None,
        google=credential if credential_provider == "google" else None,
    )

    async def transform(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["credentials"] is credentials
        value = outputs[1 if "system_prompt" in kwargs else 0]
        if value is None:
            raise RuntimeError("private-test-key https://private.invalid/query")
        return value

    async def embed(*args, **kwargs):
        return [0.1]

    async def search(*args, **kwargs):
        if search_fails:
            raise RuntimeError("private search failure")
        return [], []

    monkeypatch.setattr(query_variants, "transform_query", transform)
    monkeypatch.setattr(context, "_embed_variant", embed)
    monkeypatch.setattr(context, "_search_both", search)
    caplog.set_level(logging.WARNING)

    async def run():
        timings, token = start_request_timings()
        try:
            try:
                result = await context.get_retrieved_context_with_sources(
                    Database("postgresql://unused"),
                    "query",
                    Model.BGE_M3_LOCAL,
                    Model.GPT_5_4_MINI,
                    query_transform_mode=QueryTransformMode(mode),
                    credentials=credentials,
                )
            except RetrievalError:
                assert search_fails
            else:
                assert not search_fails
                assert result.text == ""
                assert result.effective_transform_mode.value == effective
            return timings
        finally:
            reset_request_timings(token)

    timings = anyio.run(run)
    assert timings.query_transform_fallback_reason == reason
    assert timings.reranker_fallback_reason == "not_run"
    assert timings.reranker_fallback is False
    assert timings.reranker_invalid_result_count == 0
    assert len(calls) == (len(outputs) if mode == "rewrite_hyde" else bool(outputs))
    assert "private-test-key" not in caplog.text
    assert "https://private.invalid" not in caplog.text


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        "private document",
        {"index": True, "score": 0.9},
        {"index": "0", "score": 0.9},
        {"index": 0.0, "score": 0.9},
        {"index": -1, "score": 0.9},
        {"index": 2, "score": 0.9},
        {"index": 0, "score": True},
        {"index": 0, "score": "private score"},
        {"index": 0, "score": float("nan")},
        {"index": 0, "score": float("inf")},
        {"index": 0, "score": 10**400},
    ],
)
def test_invalid_rerank_entries_do_not_hide_valid_ranking(invalid):
    ranked, count = context._valid_rerank_results(
        {"reranked_documents": [invalid, {"index": 1, "score": 0.9}]},
        2,
    )
    assert ranked == [(1, 0.9)]
    assert count == 1


@pytest.mark.parametrize("has_candidates", [False, True])
def test_real_retrieval_propagates_diagnostics_to_events_and_timing_log(
    monkeypatch,
    caplog,
    has_candidates,
):
    """Run the chat/retrieval pipeline, replacing only external I/O boundaries."""
    captured = {}
    private_query = "private query"
    faq = QuestionSchema(
        id=uuid4(),
        name="private title",
        content="private document",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        distance=0.1,
    )

    async def resolve(*args, **kwargs):
        return LlmProviderCredentials(
            openai=ChatCredentialSecret(
                provider="openai",
                api_key="private-test-key",
            )
        )

    async def transform(*args, **kwargs):
        raise RuntimeError("private error https://private.invalid")

    async def embed(*args, **kwargs):
        return [0.1]

    async def search(*args, **kwargs):
        return ([faq] if has_candidates else []), []

    class RerankResponse:
        def json(self):
            return {"reranked_documents": []}

    async def rerank(*args, **kwargs):
        return RerankResponse()

    async def links(*args, **kwargs):
        return ""

    async def lexical_search(*args, **kwargs):
        return []

    async def body():
        yield 'event: token\ndata: {"text":"answer"}\n\n'
        yield "event: done\ndata: {}\n\n"

    async def handle(*args, **kwargs):
        return SimpleNamespace(body_iterator=body())

    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve)
    monkeypatch.setattr(query_variants, "transform_query", transform)
    monkeypatch.setattr(context, "_embed_variant", embed)
    monkeypatch.setattr(context, "_search_both", search)
    monkeypatch.setattr(context, "_post_rerank", rerank)
    monkeypatch.setattr(context, "get_matching_questions", lexical_search)
    monkeypatch.setattr(chat_api, "get_links_context", links)
    monkeypatch.setattr(chat_api, "handle_chat", handle)
    monkeypatch.setattr(
        chat_api, "capture", lambda _, event, props: captured.update({event: props})
    )
    caplog.set_level(logging.INFO)
    payload = ChatSchema.model_validate(
        {
            "user_id": str(uuid4()),
            "inference_model": Model.GPT_5_6_LUNA,
            "query_transform_model": Model.GPT_5_4_MINI,
            "query_transform_mode": "rewrite_hyde",
            "embeddings_model": Model.BGE_M3_LOCAL,
            "messages": [{"role": "user", "content": private_query}],
        }
    )
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(state=SimpleNamespace(settings=settings())),
            "headers": [],
        }
    )

    async def run():
        return [
            chunk
            async for chunk in chat_api._chat_response_stream(
                payload,
                request,
                Database("postgresql://unused"),
                uuid4(),
            )
        ]

    anyio.run(run)
    for event in (
        "retrieval_used" if has_candidates else "retrieval_miss",
        "$ai_generation",
    ):
        properties = captured[event]
        assert properties["query_transform_fallback_reason"] == "no_usable_variants"
        assert properties["reranker_fallback_reason"] == (
            "empty_or_invalid_response" if has_candidates else "not_run"
        )
        assert properties["reranker_fallback"] is has_candidates
        assert properties["reranker_invalid_result_count"] == 0
        assert private_query not in repr(properties)
        assert "private document" not in repr(properties)
    assert (
        captured["$ai_generation"]["requested_query_transform_mode"] == "rewrite_hyde"
    )
    assert captured["$ai_generation"]["query_transform_mode"] == "raw"
    timing_log = next(
        record.message
        for record in caplog.records
        if record.message.startswith("chat.timing ")
    )
    assert '"query_transform_fallback_reason": "no_usable_variants"' in timing_log
    for private in (
        private_query,
        "private document",
        "private error",
        "private-test-key",
        "https://private.invalid",
    ):
        assert private not in caplog.text
        assert private not in repr(captured)
