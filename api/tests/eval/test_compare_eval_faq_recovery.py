import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.llms.faq_query_recovery import FaqQueryRecovery
from app.llms.faq_query_recovery_plan import FaqQueryRecoveryPlan
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.schemas.questions import QuestionSchema
from tests.eval import run_eval
from tests.eval.compare_eval import compare_runs, main, render_report
from tests.eval.compare_eval_models import EvalJsonError, parse_cases


def matched_runs(monkeypatch, tmp_path, initial_k):
    """Produce two real runner artifacts with only the external boundaries mocked."""
    query = "Kade e sluzba"
    faq = QuestionSchema(
        id=uuid4(),
        name="Студентска служба",
        content="ТМФ 117",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    plan = FaqQueryRecoveryPlan(
        outcome="matched",
        recovery=FaqQueryRecovery(
            title=faq.name,
            intent="location",
            canonical_query="Каде се наоѓа Студентска служба?",
            match_kind="alias",
        ),
        question=faq,
    )
    bundle = QueryVariantBundle(
        variants=(QueryVariant(kind="raw", text=query, is_document=False),),
        rerank_query=query,
    )

    class OfflineDatabase:
        async def init(self):
            return None

        async def disconnect(self):
            return None

    monkeypatch.setattr(run_eval, "Database", lambda **_kwargs: OfflineDatabase())
    monkeypatch.setattr(run_eval, "init_http_client", lambda: None)
    monkeypatch.setattr(run_eval, "close_http_client", AsyncMock())
    monkeypatch.setattr(
        run_eval, "build_query_variants", AsyncMock(return_value=bundle)
    )
    monkeypatch.setattr(
        run_eval, "plan_faq_query_recovery", AsyncMock(return_value=plan)
    )
    monkeypatch.setattr(run_eval, "_embed_variant", AsyncMock(return_value=[0.1]))
    monkeypatch.setattr(run_eval, "get_closest_questions", AsyncMock(return_value=[]))
    monkeypatch.setattr(run_eval, "get_closest_chunks", AsyncMock(return_value=[]))
    monkeypatch.setattr(run_eval, "get_matching_questions", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        run_eval,
        "_post_rerank",
        AsyncMock(
            return_value=SimpleNamespace(
                json=lambda: {"reranked_documents": [{"index": 0, "score": 0.99}]},
            ),
        ),
    )
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps(
            {"id": "public", "query": query, "anchor": {"type": "Q", "name": faq.name}}
        ),
        encoding="utf-8",
    )
    paths = [tmp_path / "baseline.json", tmp_path / "current.json"]
    for output in paths:
        ns = argparse.Namespace(
            concurrency=1,
            embedding_model=Model.BGE_M3_LOCAL.value,
            golden=str(golden),
            ideal_limit=20,
            initial_k=initial_k,
            json=str(output),
            limit=0,
            no_transform=False,
            query_transform_model=Model.GPT_5_4_MINI.value,
            top_k=10,
            transform_mode=QueryTransformMode.RAW,
        )
        assert anyio.run(run_eval.main_async, ns) == 0
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths], paths


@pytest.mark.parametrize("initial_k", [30, 100])
def test_real_matched_raw_runs_compare_with_each_other_and_themselves(
    monkeypatch,
    tmp_path,
    initial_k,
):
    (baseline, current), paths = matched_runs(monkeypatch, tmp_path, initial_k)
    assert baseline == current
    for artifact in (baseline, current):
        assert artifact["config"]["per_query_k"] == initial_k + 1
        assert artifact["results"][0]["effective_per_query_k"] == 31
        assert artifact["results"][0]["final"] is True
        parsed = parse_cases(artifact, "run")["public"]
        assert parsed.local_faq_recovery_outcome == "matched"
        assert parsed.used_requested_config is True
    for other in (baseline, current):
        comparison = compare_runs(baseline, other)
        assert comparison.incomparable_configs == []
        assert comparison.bucket_deltas["overall"][0].count == 1
        assert "Decision: PASS" in render_report(comparison)
    assert main(["--baseline", str(paths[0]), "--current", str(paths[1])]) == 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"local_faq_recovery_outcome": None},
        {"local_faq_recovery_outcome": "skipped"},
        {"local_faq_recovery_outcome": "no_match"},
        {"local_faq_recovery_outcome": "error"},
        {"effective_per_query_k": 30},
        {"effective_per_query_k": 32},
        {"effective_per_query_k": 101},
        {"effective_initial_k": 101},
        {"effective_transform_mode": "rewrite"},
    ],
)
def test_recovery_declaration_does_not_waive_other_configuration_checks(
    monkeypatch,
    tmp_path,
    mutation,
):
    (baseline, current), _paths = matched_runs(monkeypatch, tmp_path, 100)
    current["results"][0].update(mutation)
    comparison = compare_runs(baseline, current)
    assert [case.current.id for case in comparison.incomparable_configs] == ["public"]
    assert comparison.bucket_deltas["overall"][0].count == 0
    assert "Decision: INVALID" in render_report(comparison)


@pytest.mark.parametrize("effective_mode", ["raw", "rewrite"])
def test_matched_recovery_cannot_validate_a_requested_transform_mode(
    monkeypatch,
    tmp_path,
    effective_mode,
):
    (baseline, current), _paths = matched_runs(monkeypatch, tmp_path, 100)
    current["config"]["query_transform_mode"] = "rewrite"
    current["config"]["per_query_k"] = 51
    current["results"][0]["effective_transform_mode"] = effective_mode
    current["results"][0]["effective_per_query_k"] = (
        51 if effective_mode == "rewrite" else 31
    )
    assert len(compare_runs(baseline, current).incomparable_configs) == 1


@pytest.mark.parametrize("outcome", ["unknown", True, 1, {}])
def test_invalid_recovery_outcomes_are_rejected(monkeypatch, tmp_path, outcome):
    (baseline, current), _paths = matched_runs(monkeypatch, tmp_path, 100)
    current["results"][0]["local_faq_recovery_outcome"] = outcome
    with pytest.raises(EvalJsonError, match="local_faq_recovery_outcome"):
        compare_runs(baseline, current)


def test_intentional_legacy_uncapped_to_declared_recovery_comparison(
    monkeypatch,
    tmp_path,
):
    (baseline, current), _paths = matched_runs(monkeypatch, tmp_path, 100)
    baseline["results"][0].pop("local_faq_recovery_outcome")
    baseline["results"][0]["effective_per_query_k"] = 101
    assert compare_runs(baseline, current).incomparable_configs == []
    unmarked_cap = deepcopy(current)
    unmarked_cap["results"][0].pop("local_faq_recovery_outcome")
    assert len(compare_runs(baseline, unmarked_cap).incomparable_configs) == 1


def test_capped_runs_still_fail_the_gate_on_retrieval_regressions(
    monkeypatch,
    tmp_path,
):
    (baseline, current), paths = matched_runs(monkeypatch, tmp_path, 100)
    current["results"][0].update(final=False, rank=None)
    comparison = compare_runs(baseline, current)
    assert comparison.incomparable_configs == []
    assert [case.current.id for case in comparison.new_regressions] == ["public"]
    paths[1].write_text(json.dumps(current), encoding="utf-8")
    assert main(["--baseline", str(paths[0]), "--current", str(paths[1])]) == 1
