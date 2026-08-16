from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest
from asyncpg import PostgresError

from app.data.connection import Database
from app.data.questions import get_matching_questions
from app.llms import context as context_module
from app.llms.context import get_retrieved_context_with_sources
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.llms.retrieval_result import RetrievedContext
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from app.utils.timing import (
    RequestTimings,
    reset_request_timings,
    start_request_timings,
)


def _run_with_timings(run) -> tuple[RetrievedContext, RequestTimings]:
    timings, token = start_request_timings()
    try:
        return anyio.run(run), timings
    finally:
        reset_request_timings(token)


def test_lexical_faq_search_uses_weighted_or_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "id": uuid4(),
        "name": "Студиски програми",
        "content": "СИИС, SEIS и КН.",
        "user_id": None,
        "links": "{}",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    database = Database("postgresql://hybrid-test")
    fetch = AsyncMock(return_value=[row])
    monkeypatch.setattr(database, "fetch", fetch)

    async def run() -> None:
        results = await get_matching_questions(
            database,
            "smerovi SEIS23",
            limit=7,
        )

        assert [result.name for result in results] == ["Студиски програми"]

    anyio.run(run)

    assert fetch.await_args is not None
    sql, query, limit = fetch.await_args.args
    assert "websearch_to_tsquery" in sql
    assert "'simple'" in sql
    assert "setweight" in sql
    assert sql.count("setweight") == 2
    assert "lexical_document.value @@ lexical_query.value" in sql
    assert " OR " in sql
    assert query == "smerovi SEIS23"
    assert limit == 7


@pytest.mark.parametrize(
    ("vector_score", "expected_final_faq_count", "expected_retrieval_path"),
    [
        (0.05, 1, "lexical"),
        (0.5, 2, "hybrid"),
    ],
)
def test_normalized_lexical_faq_reaches_existing_reranker(
    monkeypatch: pytest.MonkeyPatch,
    vector_score: float,
    expected_final_faq_count: int,
    expected_retrieval_path: str,
) -> None:
    faq = QuestionSchema(
        id=uuid4(),
        name="Студиски програми",
        content="SEIS е студиска програма.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    reranked_documents: list[str] = []
    lexical_limits: list[int] = []
    vector_faq = QuestionSchema(
        id=uuid4(),
        name="Контакт",
        content="Контакт информации.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        distance=0.2,
    )

    async def embed(*args, **kwargs):
        return [0.1]

    async def vector_search(*args, **kwargs):
        return [vector_faq], []

    async def build_variants(*args, **kwargs):
        raw = QueryVariant(kind="raw", text="smerovi SEIS23", is_document=False)
        return QueryVariantBundle(variants=(raw,), rerank_query="rewritten query")

    async def lexical_search(_db, query, *, limit):
        lexical_limits.append(limit)
        return [vector_faq, faq] if query.startswith("смерови") else []

    class RerankResponse:
        def json(self):
            return {
                "reranked_documents": [
                    {"index": 1, "score": 0.9},
                    {"index": 0, "score": vector_score},
                ],
            }

    async def rerank(payload):
        reranked_documents.extend(payload["documents"])
        return RerankResponse()

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "build_query_variants", build_variants)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(context_module, "get_matching_questions", lexical_search)
    monkeypatch.setattr(context_module, "_post_rerank", rerank)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "smerovi SEIS23",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert result.sources[0].title == faq.name
    assert faq.content in result.text
    assert reranked_documents == [
        f"Наслов: {vector_faq.name}\nСодржина: {vector_faq.content}",
        f"Наслов: {faq.name}\nСодржина: {faq.content}",
    ]
    assert lexical_limits == [31, 31]
    assert timings.transliteration_variant_added is True
    assert timings.lexical_search_outcome == "matched"
    assert timings.dense_faq_candidate_count == 1
    assert timings.dense_document_candidate_count == 0
    assert timings.lexical_faq_candidate_count == 2
    assert timings.final_faq_count == expected_final_faq_count
    assert timings.final_document_count == 0
    assert timings.lexical_only_final_count == 1
    assert timings.retrieval_path == expected_retrieval_path
    assert timings.reranker_fallback is False


def test_lexical_faq_failure_preserves_dense_retrieval(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector_faq = QuestionSchema(
        id=uuid4(),
        name="Контакт",
        content="Контакт информации.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        distance=0.2,
    )

    async def embed(*args, **kwargs):
        return [0.1]

    async def vector_search(*args, **kwargs):
        return [vector_faq], []

    async def lexical_search(*args, **kwargs):
        raise PostgresError("lexical search unavailable")

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.9}]}

    async def rerank(*args, **kwargs):
        return RerankResponse()

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(context_module, "get_matching_questions", lexical_search)
    monkeypatch.setattr(context_module, "_post_rerank", rerank)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "kontakt na finki",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert result.sources[0].title == vector_faq.name
    assert any(record.exc_info is not None for record in caplog.records)
    assert timings.lexical_search_outcome == "error"
    assert timings.dense_faq_candidate_count == 1
    assert timings.lexical_faq_candidate_count == 0
    assert timings.final_faq_count == 1
    assert timings.retrieval_path == "dense"
    assert timings.reranker_fallback is False


def test_lexical_faq_search_is_skipped_without_vector_domain_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_calls = 0

    async def embed(*args, **kwargs):
        return [0.1]

    async def vector_search(*args, **kwargs):
        return [], []

    async def lexical_search(*args, **kwargs):
        nonlocal lexical_calls
        lexical_calls += 1
        return []

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(context_module, "get_matching_questions", lexical_search)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "рецепт за торта",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert result.text == ""
    assert lexical_calls == 0
    assert timings.lexical_search_outcome == "skipped"
    assert timings.dense_faq_candidate_count == 0
    assert timings.dense_document_candidate_count == 0
    assert timings.lexical_faq_candidate_count == 0
    assert timings.final_faq_count == 0
    assert timings.final_document_count == 0
    assert timings.lexical_only_final_count == 0
    assert timings.retrieval_path == "none"
    assert timings.reranker_fallback is False


def test_lexical_faq_search_records_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_calls = 0
    vector_faq = QuestionSchema(
        id=uuid4(),
        name="Контакт",
        content="Контакт информации.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        distance=0.2,
    )

    async def embed(*args, **kwargs):
        return [0.1]

    async def vector_search(*args, **kwargs):
        return [vector_faq], []

    async def lexical_search(*args, **kwargs):
        nonlocal lexical_calls
        lexical_calls += 1
        return []

    class RerankResponse:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.9}]}

    async def rerank(*args, **kwargs):
        return RerankResponse()

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(context_module, "get_matching_questions", lexical_search)
    monkeypatch.setattr(context_module, "_post_rerank", rerank)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "smerovi na finki",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    result, timings = _run_with_timings(run)

    assert vector_faq.content in result.text
    assert lexical_calls == 2
    assert timings.transliteration_variant_added is True
    assert timings.lexical_search_outcome == "no_match"
    assert timings.retrieval_path == "dense"


def test_reranker_failure_falls_back_to_dense_candidates_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlapping_faq = QuestionSchema(
        id=uuid4(),
        name="Студиски програми",
        content="Густ и лексички FAQ кандидат.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        distance=0.1,
    )
    lexical_only_faq = QuestionSchema(
        id=uuid4(),
        name="Изборни предмети",
        content="Само лексички FAQ кандидат.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    dense_chunk = ChunkSchema(
        id=uuid4(),
        document_id=uuid4(),
        document_name="rules",
        document_title="Правилник",
        chunk_index=0,
        content="Густо пронајден документ.",
        distance=0.2,
    )

    async def embed(*args, **kwargs):
        return [0.1]

    async def vector_search(*args, **kwargs):
        return [overlapping_faq], [dense_chunk]

    async def lexical_search(*args, **kwargs):
        return [overlapping_faq, lexical_only_faq]

    async def failing_reranker(*args, **kwargs):
        raise RuntimeError("reranker unavailable")

    async def no_neighbor_chunks(*args, **kwargs):
        return []

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(context_module, "get_matching_questions", lexical_search)
    monkeypatch.setattr(context_module, "_post_rerank", failing_reranker)
    monkeypatch.setattr(context_module, "get_chunks_window", no_neighbor_chunks)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "smerovi na finki",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
            top_k=2,
        )

    result, timings = _run_with_timings(run)

    assert overlapping_faq.content in result.text
    assert dense_chunk.content in result.text
    assert lexical_only_faq.content not in result.text
    assert timings.lexical_search_outcome == "matched"
    assert timings.dense_faq_candidate_count == 1
    assert timings.dense_document_candidate_count == 1
    assert timings.lexical_faq_candidate_count == 2
    assert timings.final_faq_count == 1
    assert timings.final_document_count == 1
    assert timings.lexical_only_final_count == 0
    assert timings.retrieval_path == "dense"
    assert timings.reranker_fallback is True
