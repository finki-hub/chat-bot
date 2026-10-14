import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.data.connection import Database
from app.llms import context as context_module
from app.llms import faq_query_recovery_plan as plan_module
from app.llms.context import _search_both, get_retrieved_context_with_sources
from app.llms.faq_query_recovery import FaqQueryRecovery
from app.llms.faq_query_recovery_plan import FaqQueryRecoveryPlan
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.timing import (
    RequestTimings,
    reset_request_timings,
    start_request_timings,
)


def _question(*, distance: float | None = None) -> QuestionSchema:
    return QuestionSchema(
        id=uuid4(),
        name="Студентска служба",
        content="Локација на FAQ.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        distance=distance,
    )


def _plan(question: QuestionSchema) -> FaqQueryRecoveryPlan:
    return FaqQueryRecoveryPlan(
        outcome="matched",
        recovery=FaqQueryRecovery(
            title=question.name,
            intent="location",
            canonical_query=f"Каде се наоѓа {question.name}?",
            match_kind="alias",
        ),
        question=question,
    )


def _run_with_timings[T](
    run: Callable[[], Awaitable[T]],
) -> tuple[T, RequestTimings]:
    timings, token = start_request_timings()
    try:
        return anyio.run(run), timings
    finally:
        reset_request_timings(token)


def _install_raw_retrieval_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: QuestionSchema,
    embedding: object,
    search: object,
    rerank: object,
) -> None:
    async def build_variants(query, *args, **kwargs):
        raw = QueryVariant(kind="raw", text=query, is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query=query)

    monkeypatch.setattr(context_module, "build_query_variants", build_variants)
    monkeypatch.setattr(context_module, "_embed_variant", embedding)
    monkeypatch.setattr(context_module, "_search_both", search)
    monkeypatch.setattr(
        context_module,
        "plan_faq_query_recovery",
        AsyncMock(return_value=_plan(question)),
    )
    monkeypatch.setattr(context_module, "_post_rerank", rerank)


def test_resolved_faq_adds_one_canonical_embedding_and_direct_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question()
    canonical = f"Каде се наоѓа {faq.name}?"
    embedded: list[str] = []
    limits: list[int] = []
    payloads: list[dict[str, object]] = []

    async def embed(text, *args, **kwargs):
        embedded.append(text)
        return [float(len(embedded))]

    async def search(_db, _embedding, _model, limit):
        limits.append(limit)
        return [], []

    async def unexpected_lexical_search(*args, **kwargs):
        raise AssertionError("resolved FAQ uses its exact candidate, not broad lexical")

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        payloads.append(payload)
        return RerankResponse()

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=search,
        rerank=rerank,
    )
    monkeypatch.setattr(
        context_module,
        "get_matching_questions",
        unexpected_lexical_search,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
            initial_k=120,
        )

    result, timings = _run_with_timings(run)

    assert faq.content in result.text
    assert [source.title for source in result.sources] == [faq.name]
    assert embedded == ["Каде е службата?", canonical]
    assert limits == [31, 31]
    assert payloads == [
        {
            "query": canonical,
            "documents": [f"Наслов: {faq.name}\nСодржина: {faq.content}"],
        },
    ]
    assert timings.local_faq_recovery_outcome == "matched"
    assert timings.local_query_variant_added is True
    assert timings.local_rerank_query_used is True


def test_resolved_faq_direct_and_vector_candidates_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question(distance=0.2)
    documents: list[str] = []

    async def embed(*args, **kwargs):
        return [0.1]

    async def search(*args, **kwargs):
        return [faq], []

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        documents.extend(payload["documents"])
        return RerankResponse()

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=search,
        rerank=rerank,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            f"Каде е {faq.name}?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert faq.content in result.text
    assert documents == [f"Наслов: {faq.name}\nСодржина: {faq.content}"]
    assert timings.dense_faq_candidate_count == 1
    assert timings.lexical_faq_candidate_count == 1
    assert timings.retrieval_path == "hybrid"


def test_resolved_faq_is_not_exposed_when_reranker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question()

    async def embed(*args, **kwargs):
        return [0.1]

    async def no_hits(*args, **kwargs):
        return [], []

    async def failed_reranker(*args, **kwargs):
        raise RuntimeError("reranker unavailable")

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=no_hits,
        rerank=failed_reranker,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert result.text == ""
    assert result.sources == ()
    assert timings.reranker_fallback_reason == "error"


def test_optional_canonical_search_failure_keeps_raw_and_direct_faq_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question()
    rerank_queries: list[str] = []
    search_embeddings: list[list[float]] = []

    async def embed(text, *args, **kwargs):
        return [1.0] if text == "Каде е службата?" else [2.0]

    async def search(_db, embedding, _model, _limit):
        search_embeddings.append(embedding)
        if embedding == [2.0]:
            raise TimeoutError("optional canonical index timed out")
        return [faq], []

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        rerank_queries.append(payload["query"])
        assert payload["documents"] == [f"Наслов: {faq.name}\nСодржина: {faq.content}"]
        return RerankResponse()

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=search,
        rerank=rerank,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert faq.content in result.text
    assert search_embeddings == [[1.0], [2.0]]
    assert rerank_queries == [f"Каде се наоѓа {faq.name}?"]
    assert timings.local_faq_recovery_outcome == "matched"
    assert timings.local_query_variant_added is True
    assert timings.local_rerank_query_used is True


@pytest.mark.parametrize("failed_source", ["faq", "chunk"])
def test_optional_canonical_search_drains_failed_search_sibling(
    monkeypatch: pytest.MonkeyPatch,
    failed_source: str,
) -> None:
    faq = _question()
    canonical_children: dict[str, asyncio.Task[object]] = {}
    cleaned: list[str] = []
    rerank_queries: list[str] = []

    async def blocked(source: str) -> None:
        task = asyncio.current_task()
        assert task is not None
        canonical_children[source] = task
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(source)

    async def get_questions(_db, embedding, _model, *, limit):
        if embedding == [1.0]:
            return [faq]
        if failed_source == "faq":
            raise TimeoutError("canonical FAQ search timed out")
        await blocked("faq")
        return []

    async def get_chunks(_db, embedding, _model, *, limit):
        if embedding == [1.0]:
            return []
        if failed_source == "chunk":
            raise TimeoutError("canonical chunk search timed out")
        await blocked("chunk")
        return []

    async def embed(text, *args, **kwargs):
        return [1.0] if text == "Каде е службата?" else [2.0]

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        rerank_queries.append(payload["query"])
        return RerankResponse()

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=AsyncMock(),
        rerank=rerank,
    )
    monkeypatch.setattr(context_module, "_search_both", _search_both)
    monkeypatch.setattr(context_module, "get_closest_questions", get_questions)
    monkeypatch.setattr(context_module, "get_closest_chunks", get_chunks)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert faq.content in result.text
    assert rerank_queries == [f"Каде се наоѓа {faq.name}?"]
    assert cleaned == ["chunk" if failed_source == "faq" else "faq"]
    assert len(canonical_children) == 1
    assert all(task.done() for task in canonical_children.values())
    assert timings.local_faq_recovery_outcome == "matched"
    assert timings.local_rerank_query_used is True


def test_search_both_cancellation_cancels_and_drains_both_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: set[str] = set()
    both_started = asyncio.Event()
    children: dict[str, asyncio.Task[object]] = {}
    cleaned: list[str] = []

    async def blocked(source: str):
        task = asyncio.current_task()
        assert task is not None
        children[source] = task
        started.add(source)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(source)

    async def get_questions(*args, **kwargs):
        return await blocked("faq")

    async def get_chunks(*args, **kwargs):
        return await blocked("chunk")

    monkeypatch.setattr(context_module, "get_closest_questions", get_questions)
    monkeypatch.setattr(context_module, "get_closest_chunks", get_chunks)

    async def run():
        search_task = asyncio.create_task(
            _search_both(
                Database("postgresql://unused"), [1.0], Model.BGE_M3_LOCAL, 31
            ),
        )
        await both_started.wait()
        search_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await search_task

    anyio.run(run)

    assert set(cleaned) == {"faq", "chunk"}
    assert set(children) == {"faq", "chunk"}
    assert all(task.done() for task in children.values())


def test_main_embedding_failure_records_match_without_claiming_rerank_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question()

    async def failed_embedding(*args, **kwargs):
        raise RuntimeError("mandatory raw embedding failed")

    async def unexpected(*args, **kwargs):
        raise AssertionError(
            "retrieval must not continue after mandatory embedding failure"
        )

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=failed_embedding,
        search=unexpected,
        rerank=unexpected,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    timings, token = start_request_timings()
    try:
        with pytest.raises(RetrievalError, match="multi-query vector search"):
            anyio.run(run)
    finally:
        reset_request_timings(token)

    assert timings.local_faq_recovery_outcome == "matched"
    assert timings.local_query_variant_added is False
    assert timings.local_rerank_query_used is False


@pytest.mark.parametrize("failed_source", ["faq", "chunk"])
@pytest.mark.parametrize(
    "stop_reason", ["raw_failure", "external_cancellation", "canonical_cancellation"]
)
def test_vector_search_failure_and_cancellation_drain_all_search_tasks(
    monkeypatch: pytest.MonkeyPatch,
    failed_source: str,
    stop_reason: str,
) -> None:
    faq = _question()
    raw_query = "Каде е службата?"
    names = AsyncMock(return_value=[faq.name])
    exact = AsyncMock(return_value=faq)
    rerank = AsyncMock(side_effect=AssertionError("stopped retrieval must not rerank"))
    parents: list[asyncio.Task[object]] = []
    children: dict[tuple[str, str], asyncio.Task[object]] = {}
    cleaned: set[tuple[str, str]] = set()
    all_started = asyncio.Event()
    raw_error = TimeoutError("mandatory raw search timed out")

    async def embed(text, *args, **kwargs):
        return [1.0] if text == raw_query else [2.0]

    async def search(db, embedding, model, limit):
        task = asyncio.current_task()
        assert task is not None
        parents.append(task)
        return await _search_both(db, embedding, model, limit)

    async def database_search(source, embedding):
        variant = "raw" if embedding == [1.0] else "canonical"
        key = (variant, source)
        task = asyncio.current_task()
        assert task is not None
        children[key] = task
        if len(children) == 4:
            all_started.set()
        try:
            await all_started.wait()
            if source == failed_source:
                if stop_reason == "raw_failure" and variant == "raw":
                    raise raw_error
                if stop_reason == "canonical_cancellation" and variant == "canonical":
                    raise asyncio.CancelledError
            await asyncio.Event().wait()
        finally:
            cleaned.add(key)
        return []

    async def questions(_db, embedding, _model, *, limit):
        return await database_search("faq", embedding)

    async def chunks(_db, embedding, _model, *, limit):
        return await database_search("chunk", embedding)

    monkeypatch.setattr(plan_module, "get_question_names_query", names)
    monkeypatch.setattr(plan_module, "get_question_by_name_query", exact)
    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", search)
    monkeypatch.setattr(context_module, "get_closest_questions", questions)
    monkeypatch.setattr(context_module, "get_closest_chunks", chunks)
    monkeypatch.setattr(context_module, "_post_rerank", rerank)

    async def run():
        with anyio.fail_after(5):
            retrieval_task = asyncio.create_task(
                get_retrieved_context_with_sources(
                    Database("postgresql://unused"),
                    raw_query,
                    Model.BGE_M3_LOCAL,
                    Model.GPT_5_4_MINI,
                    query_transform_mode=QueryTransformMode.RAW,
                ),
            )
            try:
                await all_started.wait()
                if stop_reason == "external_cancellation":
                    retrieval_task.cancel()
                if stop_reason == "raw_failure":
                    with pytest.raises(RetrievalError) as exc_info:
                        await retrieval_task
                    assert exc_info.value.__cause__ is raw_error
                else:
                    with pytest.raises(asyncio.CancelledError):
                        await retrieval_task

                # Assert before runner shutdown can clean up any leaked tasks.
                assert retrieval_task.done()
                assert len(parents) == 2
                assert len(children) == 4
                assert all(task.done() for task in parents)
                assert all(task.done() for task in children.values())
                assert cleaned == set(children)
            finally:
                # Keep a failing regression run isolated as well.
                tasks = [retrieval_task, *parents, *children.values()]
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    _, timings = _run_with_timings(run)

    names.assert_awaited_once()
    exact.assert_awaited_once()
    rerank.assert_not_awaited()
    assert timings.local_faq_recovery_outcome == "matched"
    assert timings.local_query_variant_added is False
    assert timings.local_rerank_query_used is False


def test_optional_canonical_search_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question()

    async def embed(text, *args, **kwargs):
        return [1.0] if text == "Каде е службата?" else [2.0]

    async def search(_db, embedding, _model, _limit):
        if embedding == [2.0]:
            raise asyncio.CancelledError
        return [faq], []

    async def unexpected_rerank(*args, **kwargs):
        raise AssertionError("cancelled retrieval must not rerank")

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=search,
        rerank=unexpected_rerank,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    with pytest.raises(asyncio.CancelledError):
        anyio.run(run)


def test_mandatory_raw_search_failure_remains_retrieval_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question()

    async def embed(text, *args, **kwargs):
        return [1.0] if text == "Каде е службата?" else [2.0]

    async def search(_db, embedding, _model, _limit):
        if embedding == [1.0]:
            raise RuntimeError("mandatory raw index failed")
        return [], []

    async def unexpected_rerank(*args, **kwargs):
        raise AssertionError("failed mandatory search must not rerank")

    _install_raw_retrieval_mocks(
        monkeypatch,
        question=faq,
        embedding=embed,
        search=search,
        rerank=unexpected_rerank,
    )

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    with pytest.raises(RetrievalError, match="multi-query vector search") as exc_info:
        anyio.run(run)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_partial_rewrite_mode_does_not_run_local_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_texts: list[str] = []
    rerank_queries: list[str] = []
    faq = _question(distance=0.2)
    planner = AsyncMock(side_effect=AssertionError("non-raw mode must skip recovery"))

    async def embed(text, *args, **kwargs):
        embedding_texts.append(text)
        return [0.1]

    async def search(*args, **kwargs):
        return [faq], []

    async def no_lexical_results(*args, **kwargs):
        return []

    async def build_variants(query, *args, **kwargs):
        return QueryVariantBundle(
            variants=(
                QueryVariant(kind="rewrite", text="rewritten query", is_document=False),
                QueryVariant(kind="raw", text=query, is_document=False),
            ),
            rerank_query="rewritten query",
        )

    class EmptyRerankResponse:
        def json(self):
            return {"reranked_documents": []}

    async def rerank(payload):
        rerank_queries.append(payload["query"])
        return EmptyRerankResponse()

    monkeypatch.setattr(context_module, "build_query_variants", build_variants)
    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", search)
    monkeypatch.setattr(context_module, "get_matching_questions", no_lexical_results)
    monkeypatch.setattr(context_module, "plan_faq_query_recovery", planner)
    monkeypatch.setattr(context_module, "_post_rerank", rerank)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE_HYDE,
        )

    _, timings = _run_with_timings(run)

    planner.assert_not_awaited()
    assert set(embedding_texts) == {"rewritten query", "Каде е службата?"}
    assert rerank_queries == ["rewritten query"]
    assert timings.local_faq_recovery_outcome == "skipped"
    assert timings.local_query_variant_added is False
    assert timings.local_rerank_query_used is False


def test_catalog_lookup_error_uses_original_retrieval_and_rerank_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    faq = _question(distance=0.2)
    embedding_texts: list[str] = []
    rerank_queries: list[str] = []

    async def embed(text, *args, **kwargs):
        embedding_texts.append(text)
        return [0.1]

    async def search(*args, **kwargs):
        return [faq], []

    async def no_lexical_results(*args, **kwargs):
        return []

    async def build_variants(query, *args, **kwargs):
        return QueryVariantBundle(
            variants=(QueryVariant(kind="raw", text=query, is_document=False),),
            rerank_query=query,
        )

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        rerank_queries.append(payload["query"])
        return RerankResponse()

    monkeypatch.setattr(context_module, "build_query_variants", build_variants)
    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", search)
    monkeypatch.setattr(context_module, "get_matching_questions", no_lexical_results)
    monkeypatch.setattr(
        context_module,
        "plan_faq_query_recovery",
        AsyncMock(return_value=FaqQueryRecoveryPlan(outcome="error")),
    )
    monkeypatch.setattr(context_module, "_post_rerank", rerank)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "Каде е службата?",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert faq.content in result.text
    assert embedding_texts == ["Каде е службата?"]
    assert rerank_queries == ["Каде е службата?"]
    assert timings.local_faq_recovery_outcome == "error"
    assert timings.local_query_variant_added is False
    assert timings.local_rerank_query_used is False


def test_recovery_uses_contextualized_query_and_rejects_foreign_modifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_query = "Каде е службата?"
    search_query = "Каде е студентската служба на Оксфорд?"
    names = AsyncMock(return_value=["Студентска служба"])
    exact = AsyncMock()
    embedded: list[str] = []
    planned_queries: list[str] = []
    monkeypatch.setattr(plan_module, "get_question_names_query", names)
    monkeypatch.setattr(plan_module, "get_question_by_name_query", exact)
    monkeypatch.setattr(context_module, "has_provider_credential", lambda *args: True)

    async def contextualize(*args, **kwargs):
        return search_query

    async def plan(db, query):
        planned_queries.append(query)
        return await plan_module.plan_faq_query_recovery(db, query)

    async def embed(text, *args, **kwargs):
        embedded.append(text)
        return [0.1]

    async def variants(query, *args, **kwargs):
        return QueryVariantBundle(
            variants=(QueryVariant(kind="raw", text=query, is_document=False),),
            rerank_query=query,
        )

    async def no_hits(*args, **kwargs):
        return [], []

    async def no_lexical_results(*args, **kwargs):
        return []

    monkeypatch.setattr(context_module, "_contextualize_query", contextualize)
    monkeypatch.setattr(context_module, "plan_faq_query_recovery", plan)
    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "build_query_variants", variants)
    monkeypatch.setattr(context_module, "_search_both", no_hits)
    monkeypatch.setattr(context_module, "get_matching_questions", no_lexical_results)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            original_query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
            history_text="User referred to a different university.",
        )

    result, timings = _run_with_timings(run)

    assert result.text == ""
    assert embedded == [search_query]
    assert planned_queries == [search_query]
    names.assert_awaited_once()
    assert names.await_args is not None
    assert isinstance(names.await_args.args[0], Database)
    assert exact.await_count == 0
    assert timings.local_faq_recovery_outcome == "no_match"
    assert timings.local_query_variant_added is False
    assert timings.local_rerank_query_used is False
