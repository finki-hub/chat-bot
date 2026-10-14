from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.eval.faq_evidence_scope_checks import capture_scope_result
from tests.eval.faq_evidence_scope_compare import compare_scope
from tests.eval.robust_faq_compare import PAIRED_MANIFEST_KEYS


def _inputs():
    case = {
        "id": "guard",
        "family": "scope",
        "split": "offline",
        "expectation": "scope_guard_required",
        "forbidden_titles": ["Office"],
        "forbidden_fact_keys": ["location"],
    }
    office = SimpleNamespace(
        key="Q:a",
        retrieval_source=SimpleNamespace(id="a", kind="faq", title="Office"),
        distance=0.2,
        retrieval_path="dense",
    )
    manifest: dict = dict.fromkeys(PAIRED_MANIFEST_KEYS, "same")
    manifest["evaluation_source_sha256"] = {"runner": "same"}
    manifest.update(
        fixture_sha256="fixture",
        scope_catalog_sha256="catalog",
        source_sha256={"llms/context.py": "source"},
    )
    runs = []
    for hook in (False, True):
        observation = capture_scope_result(
            before=[office],
            after=[] if hook else [office],
            selected_ids=[] if hook else [office.key],
            sources=[] if hook else [office.retrieval_source],
            context_text="" if hook else "Public location.",
            diagnostics={
                "active": True,
                "strict_abstention": True,
                "reason": "institution_scope",
                "excluded_candidate_count": 1,
                "constrained_abstained": True,
            }
            if hook
            else {},
            hook_available=hook,
            case=case,
            catalog={"facts": {"location": {"quote": "Public location."}}},
        )
        row = {
            **case,
            "execution_status": "completed",
            "effective_transform_mode": "raw",
            "n_candidates": len(observation["after_candidates"]),
            "candidates": observation["after_candidates"],
            "selected_ids": observation["selected_ids"],
            "visible_sources": observation["visible_sources"],
            "context_sha256": observation["context_sha256"],
            "local_recovery_activated": False,
            "forbidden_context_hit": False,
            "scope_observation": observation,
            "db_read_failures": {"total": 0, "by_error_type": {}},
            "fallback": {
                "faq_evidence_constraint_applied": True,
                "faq_evidence_constraint_reason": "institution_scope",
                "faq_evidence_excluded_candidate_count": 1,
                "faq_evidence_constrained_abstained": True,
            }
            if hook
            else {},
        }
        runs.append(
            {
                "schema_version": "robust-faq-replay-v1",
                "status": "completed",
                "manifest": deepcopy(manifest),
                "preflight": {"valid": True},
                "db_read_failures": {"total": 0, "by_error_type": {}},
                "gpu_http_failures": {},
                "results": [row],
            }
        )
    return *runs, [case]


def test_scope_pair_validated_actual_removal_passes():
    before, after, cases = _inputs()
    report = compare_scope(
        before, after, cases, fixture_hash="fixture", catalog_hash="catalog"
    )
    assert report["valid"]
    assert report["scope_gate_pass"]
    assert report["results"][0]["baseline_forbidden"]["forbidden_fact_hits"] == [
        "location"
    ]
    assert (
        "release_gate_pass" not in report
    )  # Separate P1 preservation pair still required.


@pytest.mark.parametrize(
    "failure",
    [
        "db",
        "gpu",
        "gpu_missing",
        "catalog",
        "fixture",
        "source_runtime",
        "pool",
        "diagnostics",
        "forbidden_ledger",
    ],
)
def test_scope_validity_faults_are_not_denominator_reductions(failure):
    before, after, cases = _inputs()
    if failure == "db":
        after["db_read_failures"]["total"] = 1
    elif failure == "gpu":
        after["gpu_http_failures"]["/rerank/"] = 1
    elif failure == "gpu_missing":
        del after["gpu_http_failures"]
    elif failure in {"catalog", "fixture"}:
        after["manifest"][
            "scope_catalog_sha256" if failure == "catalog" else "fixture_sha256"
        ] = "changed"
    elif failure == "source_runtime":
        after["manifest"]["gpu_manifest_sha256"] = "changed"
    elif failure == "pool":
        after["results"][0]["scope_observation"]["after_candidates"] = before[
            "results"
        ][0]["candidates"]
    elif failure == "diagnostics":
        after["results"][0]["scope_observation"]["diagnostics"][
            "excluded_candidate_count"
        ] = 0
    else:
        before["results"][0]["scope_observation"]["forbidden_selected_keys"] = []
    with pytest.raises(
        ValueError, match=r"failure|hash|Incomparable|mismatch|incomplete"
    ):
        compare_scope(
            before, after, cases, fixture_hash="fixture", catalog_hash="catalog"
        )
