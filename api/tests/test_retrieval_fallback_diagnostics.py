import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import anyio
import pytest
from fastapi import Request

from app.api import chat as chat_api
from app.data.connection import Database
from app.llms import context
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.schemas.chat import ChatSchema
from app.schemas.chat_credentials import ChatCredentialSecret
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.timing import reset_request_timings, start_request_timings
from tests.chat_models_access_support import settings


@pytest.mark.parametrize("search_fails", [False, True])
def test_reranker_diagnostics_survive_no_candidates_and_search_errors(
    *,
    monkeypatch,
    caplog,
    search_fails,
):
    async def embed(*args, **kwargs):
        return [0.1]

    async def search(*args, **kwargs):
        if search_fails:
            raise RuntimeError("private search failure https://private.invalid")
        return [], []

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
                )
            except RetrievalError:
                assert search_fails
            else:
                assert not search_fails
                assert result.text == ""
            return timings
        finally:
            reset_request_timings(token)

    timings = anyio.run(run)
    assert timings.reranker_fallback_reason == "not_run"
    assert timings.reranker_fallback is False
    assert timings.reranker_invalid_result_count == 0
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
    """Run the chat endpoint with the agent's search, replacing only external I/O."""
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

    async def embed(*args, **kwargs):
        return [0.1]

    async def search(*args, **kwargs):
        return ([faq] if has_candidates else []), []

    class RerankResponse:
        def json(self):
            return {"reranked_documents": []}

    async def rerank(*args, **kwargs):
        return RerankResponse()

    async def lexical_search(*args, **kwargs):
        return []

    async def body():
        # Stands in for the agent calling search_knowledge_base mid-stream.
        await context.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            private_query,
            Model.BGE_M3_LOCAL,
        )
        yield 'event: token\ndata: {"text":"answer"}\n\n'
        yield "event: done\ndata: {}\n\n"

    async def handle(*args, **kwargs):
        return SimpleNamespace(body_iterator=body())

    monkeypatch.setattr(chat_api, "resolve_provider_credentials", resolve)
    monkeypatch.setattr(context, "_embed_variant", embed)
    monkeypatch.setattr(context, "_search_both", search)
    monkeypatch.setattr(context, "_post_rerank", rerank)
    monkeypatch.setattr(context, "get_matching_questions", lexical_search)
    monkeypatch.setattr(chat_api, "handle_chat", handle)
    monkeypatch.setattr(
        chat_api, "capture", lambda _, event, props: captured.update({event: props})
    )
    caplog.set_level(logging.INFO)
    payload = ChatSchema.model_validate(
        {
            "user_id": str(uuid4()),
            "inference_model": Model.GPT_6_LUNA,
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
    properties = captured["$ai_generation"]
    assert properties["reranker_fallback_reason"] == (
        "empty_or_invalid_response" if has_candidates else "not_run"
    )
    assert properties["reranker_fallback"] is has_candidates
    assert properties["reranker_invalid_result_count"] == 0
    assert "query_transform_fallback_reason" not in properties
    timing_log = next(
        record.message
        for record in caplog.records
        if record.message.startswith("chat.timing ")
    )
    assert '"reranker_fallback": %s' % ("true" if has_candidates else "false") in (
        timing_log
    )
    for private in (
        private_query,
        "private document",
        "private-test-key",
        "https://private.invalid",
    ):
        assert private not in caplog.text
        assert private not in repr(captured)
