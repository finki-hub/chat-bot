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
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema


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


def test_normalized_lexical_faq_reaches_existing_reranker(
    monkeypatch: pytest.MonkeyPatch,
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
        return [faq] if query.startswith("смерови") else []

    class RerankResponse:
        def json(self):
            return {
                "reranked_documents": [
                    {"index": 1, "score": 0.9},
                    {"index": 0, "score": 0.05},
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

    result = anyio.run(run)

    assert result.sources[0].title == faq.name
    assert faq.content in result.text
    assert reranked_documents == [
        f"Наслов: {vector_faq.name}\nСодржина: {vector_faq.content}",
        f"Наслов: {faq.name}\nСодржина: {faq.content}",
    ]
    assert lexical_limits == [31, 31]


def test_lexical_faq_failure_preserves_dense_retrieval(
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

    result = anyio.run(run)

    assert result.sources[0].title == vector_faq.name


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

    result = anyio.run(run)

    assert result.text == ""
    assert lexical_calls == 0


def test_reranker_failure_falls_back_to_dense_candidates_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lexical_faq = QuestionSchema(
        id=uuid4(),
        name="Студиски програми",
        content="Лексички FAQ кандидат.",
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
        return [], [dense_chunk]

    async def lexical_search(*args, **kwargs):
        return [lexical_faq]

    async def failing_reranker(*args, **kwargs):
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", vector_search)
    monkeypatch.setattr(context_module, "get_matching_questions", lexical_search)
    monkeypatch.setattr(context_module, "_post_rerank", failing_reranker)

    async def run():
        return await get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            "smerovi na finki",
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
            top_k=1,
        )

    result = anyio.run(run)

    assert dense_chunk.content in result.text
    assert lexical_faq.content not in result.text
