import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.data.connection import Database
from app.llms.faq_query_recovery import FaqQueryRecovery
from app.llms.faq_query_recovery_plan import FaqQueryRecoveryPlan
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.schemas.questions import QuestionSchema
from tests.eval import run_eval


@pytest.mark.parametrize(
    ("canonical_embedding_fails", "canonical_search_fails"),
    [(False, False), (True, False), (False, True)],
)
def test_shared_recovery_recovers_exact_faq_and_uses_canonical_rerank(
    monkeypatch, canonical_embedding_fails, canonical_search_fails
):
    query = "Kade e sluzba"
    canonical = "Каде се наоѓа студентската служба?"
    faq = QuestionSchema(
        id=uuid4(),
        name="Студентска служба",
        content="ТМФ 117",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    recovery = FaqQueryRecovery(
        title=faq.name, intent="location", canonical_query=canonical, match_kind="alias"
    )
    planner = AsyncMock(
        return_value=FaqQueryRecoveryPlan(
            outcome="matched", recovery=recovery, question=faq
        )
    )
    embeddings = []

    async def embed(text, *_args, **_kwargs):
        embeddings.append(text)
        if canonical_embedding_fails and text == canonical:
            raise ValueError("synthetic failure")
        return [0.2] if text == canonical else [0.1]

    async def search(_db, embedding, *_args, **kwargs):
        if (
            canonical_search_fails
            and embedding == [0.2]
            and kwargs.get("threshold") is None
        ):
            raise ValueError("synthetic optional search failure")
        return []

    empty = AsyncMock(side_effect=search)
    lexical = AsyncMock(return_value=[])
    rerank = AsyncMock(
        return_value=SimpleNamespace(
            json=lambda: {"reranked_documents": [{"index": 0, "score": 0.99}]}
        )
    )
    bundle = QueryVariantBundle(
        variants=(QueryVariant(kind="raw", text=query, is_document=False),),
        rerank_query=query,
    )
    monkeypatch.setattr(
        run_eval, "build_query_variants", AsyncMock(return_value=bundle)
    )
    monkeypatch.setattr(run_eval, "plan_faq_query_recovery", planner)
    monkeypatch.setattr(run_eval, "_embed_variant", embed)
    monkeypatch.setattr(run_eval, "get_closest_questions", empty)
    monkeypatch.setattr(run_eval, "get_closest_chunks", empty)
    monkeypatch.setattr(run_eval, "get_matching_questions", lexical)
    monkeypatch.setattr(run_eval, "_post_rerank", rerank)
    db = Database("postgresql://unused")

    async def run():
        return await run_eval.evaluate_one(
            db,
            run_eval.Example(
                id="public", query=query, anchor={"type": "Q", "name": faq.name}
            ),
            embedding_model=Model.BGE_M3_LOCAL,
            qt_model=Model.GPT_5_4_MINI,
            initial_k=100,
            top_k=10,
            ideal_limit=20,
            transform_mode=QueryTransformMode.RAW,
        )

    result = anyio.run(run)
    planner.assert_awaited_once_with(db, query)
    lexical.assert_not_awaited()
    assert embeddings == [query, canonical]
    assert result.ann_prod is False
    assert result.candidate_hit is True
    assert result.final is True
    assert result.n_candidates == 1
    assert result.effective_per_query_k == 31
    assert result.local_faq_recovery_outcome == "matched"
    assert result.local_query_variant_added is (
        not canonical_embedding_fails and not canonical_search_fails
    )
    assert result.local_rerank_query_used is True
    assert rerank.await_args is not None
    assert rerank.await_args.args[0]["query"] == canonical
    assert all(call.kwargs["limit"] == 31 for call in empty.await_args_list)


def test_transformed_bundle_does_not_call_local_recovery_planner(monkeypatch):
    query = "query"
    bundle = QueryVariantBundle(
        variants=(
            QueryVariant(kind="rewrite", text="rewritten", is_document=False),
            QueryVariant(kind="raw", text=query, is_document=False),
        ),
        rerank_query="rewritten",
    )
    planner = AsyncMock()
    monkeypatch.setattr(
        run_eval, "build_query_variants", AsyncMock(return_value=bundle)
    )
    monkeypatch.setattr(run_eval, "plan_faq_query_recovery", planner)
    monkeypatch.setattr(run_eval, "_embed_variant", AsyncMock(return_value=[0.1]))
    monkeypatch.setattr(run_eval, "get_closest_questions", AsyncMock(return_value=[]))
    monkeypatch.setattr(run_eval, "get_closest_chunks", AsyncMock(return_value=[]))

    async def run():
        return await run_eval.evaluate_one(
            Database("postgresql://unused"),
            run_eval.Example(id="public", query=query, anchor={"type": "none"}),
            embedding_model=Model.BGE_M3_LOCAL,
            qt_model=Model.GPT_5_4_MINI,
            initial_k=30,
            top_k=10,
            ideal_limit=20,
            transform_mode=QueryTransformMode.REWRITE,
        )

    result = anyio.run(run)
    planner.assert_not_awaited()
    assert result.local_faq_recovery_outcome == "skipped"
    assert result.local_query_variant_added is False
    assert result.local_rerank_query_used is False
    assert result.effective_transform_mode == QueryTransformMode.REWRITE


def test_ann_pair_failure_drains_sibling_and_preserves_original_error(monkeypatch):
    drained = []

    async def questions(*_args, **_kwargs):
        raise ValueError("original read failure")

    async def chunks(*_args, **_kwargs):
        future: asyncio.Future[None] = asyncio.Future()
        try:
            await future
        finally:
            drained.append(True)
        return []

    monkeypatch.setattr(run_eval, "get_closest_questions", questions)
    monkeypatch.setattr(run_eval, "get_closest_chunks", chunks)

    async def run():
        before = set(asyncio.all_tasks())
        with pytest.raises(ValueError, match="original read failure"):
            await run_eval._search_pair(
                Database("postgresql://unused"), [0.1], Model.BGE_M3_LOCAL, limit=31
            )
        assert set(asyncio.all_tasks()) == before

    anyio.run(run)
    assert drained == [True]
