from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.data.connection import Database
from app.llms import context
from app.llms.faq_evidence_constraints import (
    FaqEvidenceConstraintPlan,
    parse_explicit_faq_constraints,
)
from app.llms.faq_query_recovery_plan import FaqQueryRecoveryPlan
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.schemas.questions import QuestionSchema
from app.utils import timing
from tests.eval import robust_faq_replay as replay
from tests.eval import run_eval


def _faq(title):
    return QuestionSchema(
        id=uuid4(),
        name=title,
        content=f"Public content for {title}",
        links={},
        distance=0.2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _plan(query):
    return FaqEvidenceConstraintPlan(
        outcome="matched",
        constraints=parse_explicit_faq_constraints(
            query,
            [
                "Студентска служба",
                "Деканат",
                "Архива",
                "Контакт и локација",
                "Компјутерски центар",
            ],
        ),
    )


@pytest.mark.parametrize(
    "mode", [QueryTransformMode.RAW, QueryTransformMode.REWRITE_HYDE]
)
@pytest.mark.parametrize("hook_available", [True, False])
@pytest.mark.parametrize("score", [0.9, 0.05])
def test_actual_observer_pool_is_filtered_before_rerank(
    monkeypatch, mode, hook_available, score
):
    query = "Каде е студентската служба на Медицински факултет?"
    service, process = _faq("Студентска служба"), _faq("Признавање кредити")
    planner = AsyncMock(
        return_value=_plan(query)
        if hook_available
        else FaqEvidenceConstraintPlan(outcome="none")
    )
    if not hook_available:
        monkeypatch.delattr(context, "apply_faq_constraints_to_candidates")
    monkeypatch.setattr(context, "plan_faq_evidence_constraints", planner)
    monkeypatch.setattr(
        context,
        "plan_faq_query_recovery",
        AsyncMock(return_value=FaqQueryRecoveryPlan(outcome="skipped")),
    )
    monkeypatch.setattr(context, "_embed_variant", AsyncMock(return_value=[1.0]))
    monkeypatch.setattr(
        context, "_search_both", AsyncMock(return_value=([service, process], []))
    )
    monkeypatch.setattr(context, "get_matching_questions", AsyncMock(return_value=[]))
    rerank = AsyncMock(
        return_value=SimpleNamespace(
            json=lambda: {"reranked_documents": [{"index": 0, "score": score}]}
        )
    )
    monkeypatch.setattr(context, "_post_rerank", rerank)

    async def render(_db, final, _model):
        return "\n".join(candidate.context_text for candidate in final)

    monkeypatch.setattr(context, "_expand_and_render", render)
    db = SimpleNamespace(fetch=AsyncMock(return_value=[]), exact_title_reads=[])
    case = {
        "id": "guard",
        "family": "foreign_office",
        "split": "offline",
        "expectation": "scope_guard_required",
        "query": query,
        "forbidden_titles": [service.name],
        "forbidden_fact_keys": [],
    }

    async def run():
        return await replay.observe_case(
            context,
            timing,
            Model,
            case,
            anchor=None,
            db=db,
            mode=mode,
            scope_catalog={"facts": {}},
        )

    row = anyio.run(run)
    assert row["execution_status"] == "completed"
    assert len(row["scope_observation"]["before_candidates"]) == 2
    assert row["n_candidates"] == (1 if hook_available else 2)
    assert row["candidates"][0]["title"] == (
        process.name if hook_available else service.name
    )
    assert row["scope_observation"]["excluded_keys"] == (
        [f"Q:{service.id}"] if hook_available else []
    )
    if hook_available:
        assert row["scope_observation"]["diagnostics"]["excluded_candidate_count"] == 1
        assert row["scope_observation"]["diagnostics"]["active"] is True
    else:
        assert row["scope_observation"]["diagnostics"] == {}
        assert (
            row["scope_observation"]["before_candidates"]
            == row["scope_observation"]["after_candidates"]
        )
    assert rerank.await_args is not None
    assert len(rerank.await_args.args[0]["documents"]) == (1 if hook_available else 2)
    assert row["selected_ids"] == (
        []
        if hook_available and score < 0.1
        else [f"Q:{process.id if hook_available else service.id}"]
    )
    if hook_available:
        assert row["scope_observation"]["diagnostics"]["constrained_abstained"] is (
            score < 0.1
        )
    planner.assert_awaited_once_with(db, query)


@pytest.mark.parametrize(
    "mode", [QueryTransformMode.RAW, QueryTransformMode.REWRITE_HYDE]
)
@pytest.mark.parametrize(
    "rerank_result", ["below_floor", "invalid", "failure", "above_floor"]
)
def test_legacy_shared_original_constraints_and_strict_abstention(
    monkeypatch, mode, rerank_result
):
    query = "Каде е деканатот, не прашувам за студентската служба?"
    service, dean = _faq("Студентска служба"), _faq("Деканат")
    plan = _plan(query)
    assert plan.constraints is not None
    planner = AsyncMock(return_value=plan)
    monkeypatch.setattr(run_eval, "plan_faq_evidence_constraints", planner)
    monkeypatch.setattr(
        run_eval,
        "plan_faq_query_recovery",
        AsyncMock(return_value=FaqQueryRecoveryPlan(outcome="skipped")),
    )
    # A rewrite drops the exclusion: the planner must still see the original.
    variants: tuple[QueryVariant, ...] = (
        QueryVariant(kind="raw", text=query, is_document=False),
    )
    if mode == QueryTransformMode.REWRITE_HYDE:
        variants += (
            QueryVariant(kind="rewrite", text="Каде е деканатот?", is_document=False),
            QueryVariant(
                kind="hyde",
                text="Деканатот е во административната зграда.",
                is_document=True,
            ),
        )
    bundle = QueryVariantBundle(
        variants=variants,
        rerank_query="Каде е деканатот?",
    )
    monkeypatch.setattr(
        run_eval, "build_query_variants", AsyncMock(return_value=bundle)
    )
    monkeypatch.setattr(run_eval, "_embed_variant", AsyncMock(return_value=[1.0]))
    monkeypatch.setattr(
        run_eval, "get_closest_questions", AsyncMock(return_value=[service, dean])
    )
    monkeypatch.setattr(run_eval, "get_closest_chunks", AsyncMock(return_value=[]))
    monkeypatch.setattr(run_eval, "get_matching_questions", AsyncMock(return_value=[]))
    if rerank_result == "failure":
        rerank = AsyncMock(side_effect=ValueError("offline failure"))
    else:
        index = 999 if rerank_result == "invalid" else 0
        score = 0.05 if rerank_result == "below_floor" else 0.9
        rerank = AsyncMock(
            return_value=SimpleNamespace(
                json=lambda: {"reranked_documents": [{"index": index, "score": score}]}
            )
        )
    monkeypatch.setattr(run_eval, "_post_rerank", rerank)
    db = Database("postgresql://unused")

    async def run():
        return await run_eval.evaluate_one(
            db,
            run_eval.Example(
                id="public", query=query, anchor={"type": "Q", "name": dean.name}
            ),
            embedding_model=Model.BGE_M3_LOCAL,
            qt_model=Model.GPT_5_4_MINI,
            initial_k=30,
            top_k=10,
            ideal_limit=30,
            transform_mode=mode,
        )

    result = anyio.run(run)
    planner.assert_awaited_once_with(db, query)
    assert result.n_candidates == 1
    assert result.faq_evidence_excluded_candidate_count == 1
    assert result.faq_evidence_constraint_reason == "entity_exclusion"
    assert result.final is (rerank_result == "above_floor")
    assert result.faq_evidence_constrained_abstained is (rerank_result != "above_floor")
    assert rerank.await_args is not None
    assert len(rerank.await_args.args[0]["documents"]) == 1


@pytest.mark.parametrize(
    "query",
    [
        "Каде е студентската служба на ФИНКИ денес?",
        "Објасни го цитатот „не прашувам за студентска служба“.",
    ],
)
def test_added_offline_modifier_and_quoted_marker_controls_do_not_change_frozen_fixture(
    query,
):
    assert parse_explicit_faq_constraints(query) is None
