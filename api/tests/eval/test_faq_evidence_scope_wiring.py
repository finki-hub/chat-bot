from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
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


@pytest.mark.parametrize("hook_available", [True, False])
@pytest.mark.parametrize("score", [0.9, 0.01])
def test_actual_observer_pool_is_filtered_before_rerank(
    monkeypatch, hook_available, score
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
    "rerank_result", ["below_floor", "invalid", "failure", "above_floor"]
)
def test_legacy_shared_original_constraints_and_strict_abstention(
    monkeypatch, rerank_result
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
    bundle = QueryVariantBundle(
        variants=(QueryVariant(kind="raw", text=query, is_document=False),),
        rerank_query=query,
    )
    monkeypatch.setattr(run_eval, "build_query_variants", Mock(return_value=bundle))
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
        score = 0.01 if rerank_result == "below_floor" else 0.9
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
            initial_k=30,
            top_k=10,
            ideal_limit=30,
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


@pytest.mark.parametrize(
    ("stage", "abstained"),
    [
        ("empty_retrieval", False),
        ("filtered_all", True),
        ("below_floor", True),
        ("invalid", True),
        ("failure", True),
        ("empty_response", True),
        ("other_target", False),
        ("top_k_zero", False),
    ],
)
def test_actual_production_and_legacy_abstention_denial_stage_parity(
    monkeypatch,
    stage,
    abstained,
):
    query = "Kade e studentska sluzba na Praven fakultet?"
    plan = _plan(query)
    assert plan.constraints is not None
    remaining = _faq("Признавање кредити")
    questions = (
        []
        if stage == "empty_retrieval"
        else [_faq("Студентска служба")]
        if stage == "filtered_all"
        else [remaining]
    )
    bundle = QueryVariantBundle(
        variants=(QueryVariant(kind="raw", text=query, is_document=False),),
        rerank_query=query,
    )
    top_k = 0 if stage == "top_k_zero" else 10
    for module in (context, run_eval):
        monkeypatch.setattr(
            module, "plan_faq_evidence_constraints", AsyncMock(return_value=plan)
        )
        monkeypatch.setattr(
            module,
            "plan_faq_query_recovery",
            AsyncMock(return_value=FaqQueryRecoveryPlan(outcome="skipped")),
        )
        monkeypatch.setattr(module, "build_query_variants", Mock(return_value=bundle))
        monkeypatch.setattr(module, "_embed_variant", AsyncMock(return_value=[1.0]))
        monkeypatch.setattr(
            module, "get_matching_questions", AsyncMock(return_value=[])
        )
    monkeypatch.setattr(
        context, "_search_both", AsyncMock(return_value=(questions, []))
    )
    monkeypatch.setattr(
        run_eval, "get_closest_questions", AsyncMock(return_value=questions)
    )
    monkeypatch.setattr(run_eval, "get_closest_chunks", AsyncMock(return_value=[]))
    reranks = []
    for module in (context, run_eval):
        if stage == "failure":
            rerank = AsyncMock(side_effect=ValueError("offline failure"))
        else:
            ranked = (
                []
                if stage == "empty_response"
                else [
                    {
                        "index": 999 if stage == "invalid" else 0,
                        "score": 0.01 if stage == "below_floor" else 0.9,
                    }
                ]
            )
            rerank = AsyncMock(
                return_value=SimpleNamespace(
                    json=lambda ranked=ranked: {"reranked_documents": ranked}
                )
            )
        reranks.append(rerank)
        monkeypatch.setattr(module, "_post_rerank", rerank)

    async def render(_db, final, _model):
        return "\n".join(candidate.context_text for candidate in final)

    monkeypatch.setattr(context, "_expand_and_render", render)
    db = Database("postgresql://unused")

    async def run():
        observed, token = timing.start_request_timings()
        try:
            production = await context.get_retrieved_context_with_sources(
                db,
                query,
                Model.BGE_M3_LOCAL,
                top_k=top_k,
                credentials=None,
            )
            diagnostics = observed.fallback_record()
        finally:
            timing.reset_request_timings(token)
        legacy = await run_eval.evaluate_one(
            db,
            run_eval.Example(
                id="offline",
                query=query,
                anchor={
                    "type": "Q",
                    "name": "Пракса" if stage == "other_target" else remaining.name,
                },
            ),
            embedding_model=Model.BGE_M3_LOCAL,
            initial_k=30,
            top_k=top_k,
            ideal_limit=30,
        )
        return production, diagnostics, legacy

    production, diagnostics, legacy = anyio.run(run)
    assert (
        diagnostics["faq_evidence_constraint_applied"]
        is legacy.faq_evidence_constraint_applied
        is True
    )
    assert diagnostics["faq_evidence_constrained_abstained"] is abstained
    assert legacy.faq_evidence_constrained_abstained is abstained
    assert (
        diagnostics["faq_evidence_excluded_candidate_count"]
        == legacy.faq_evidence_excluded_candidate_count
        == (1 if stage == "filtered_all" else 0)
    )
    assert legacy.n_candidates == (
        0 if stage in {"empty_retrieval", "filtered_all"} else 1
    )
    assert legacy.final is False
    assert bool(production.text) is (stage == "other_target")
    for rerank in reranks:
        if stage in {"empty_retrieval", "filtered_all"}:
            rerank.assert_not_awaited()
        else:
            rerank.assert_awaited_once()


@pytest.mark.parametrize("rerank_result", ["below_floor", "failure"])
def test_ordinary_legacy_rerank_handling_is_not_changed(monkeypatch, rerank_result):
    query = "ordinary request"
    faq = _faq("Признавање кредити")
    bundle = QueryVariantBundle(
        variants=(QueryVariant(kind="raw", text=query, is_document=False),),
        rerank_query=query,
    )
    monkeypatch.setattr(
        run_eval,
        "plan_faq_evidence_constraints",
        AsyncMock(return_value=FaqEvidenceConstraintPlan(outcome="none")),
    )
    monkeypatch.setattr(
        run_eval,
        "plan_faq_query_recovery",
        AsyncMock(return_value=FaqQueryRecoveryPlan(outcome="skipped")),
    )
    monkeypatch.setattr(run_eval, "build_query_variants", Mock(return_value=bundle))
    monkeypatch.setattr(run_eval, "_embed_variant", AsyncMock(return_value=[1.0]))
    monkeypatch.setattr(
        run_eval, "get_closest_questions", AsyncMock(return_value=[faq])
    )
    monkeypatch.setattr(run_eval, "get_closest_chunks", AsyncMock(return_value=[]))
    monkeypatch.setattr(run_eval, "get_matching_questions", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        run_eval,
        "_post_rerank",
        AsyncMock(side_effect=ValueError("offline failure"))
        if rerank_result == "failure"
        else AsyncMock(
            return_value=SimpleNamespace(
                json=lambda: {"reranked_documents": [{"index": 0, "score": 0.01}]}
            )
        ),
    )

    async def run():
        return await run_eval.evaluate_one(
            Database("postgresql://unused"),
            run_eval.Example(
                id="ordinary", query=query, anchor={"type": "Q", "name": faq.name}
            ),
            embedding_model=Model.BGE_M3_LOCAL,
            initial_k=30,
            top_k=10,
            ideal_limit=30,
        )

    if rerank_result == "failure":
        with pytest.raises(ValueError, match="offline failure"):
            anyio.run(run)
    else:
        result = anyio.run(run)
        assert result.final is True  # Existing unconstrained keep-best fallback.
        assert result.faq_evidence_constraint_applied is False
        assert result.faq_evidence_constrained_abstained is False


def test_catalog_failure_remains_production_abstention_and_legacy_execution_error(
    monkeypatch,
):
    query = "Kade e studentska sluzba na Praven fakultet?"
    bundle = QueryVariantBundle(
        variants=(QueryVariant(kind="raw", text=query, is_document=False),),
        rerank_query=query,
    )
    legacy_build = Mock(return_value=bundle)
    legacy_embed = AsyncMock(return_value=[1.0])
    for module in (context, run_eval):
        monkeypatch.setattr(
            module,
            "plan_faq_evidence_constraints",
            AsyncMock(return_value=FaqEvidenceConstraintPlan(outcome="error")),
        )
        monkeypatch.setattr(
            module,
            "build_query_variants",
            legacy_build if module is run_eval else Mock(return_value=bundle),
        )
        monkeypatch.setattr(
            module,
            "_embed_variant",
            legacy_embed if module is run_eval else AsyncMock(return_value=[1.0]),
        )
    db = Database("postgresql://unused")

    async def run():
        with pytest.raises(ValueError, match="constraint catalog failure"):
            await run_eval.evaluate_one(
                db,
                run_eval.Example(
                    id="catalog-error", query=query, anchor={"type": "none"}
                ),
                embedding_model=Model.BGE_M3_LOCAL,
                initial_k=30,
                top_k=10,
                ideal_limit=30,
            )
        legacy_build.assert_not_called()
        legacy_embed.assert_not_awaited()
        observed, token = timing.start_request_timings()
        try:
            production = await context.get_retrieved_context_with_sources(
                db,
                query,
                Model.BGE_M3_LOCAL,
                credentials=None,
            )
            assert production.text == ""
            assert (
                observed.fallback_record()["faq_evidence_constrained_abstained"] is True
            )
            assert (
                observed.fallback_record()["faq_evidence_constraint_reason"]
                == "catalog_error"
            )
        finally:
            timing.reset_request_timings(token)

    anyio.run(run)
