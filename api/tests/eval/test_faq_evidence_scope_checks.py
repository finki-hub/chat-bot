from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.eval.faq_evidence_scope_checks import (
    capture_scope_result,
    compare_scope_case,
    validate_scope_observation,
)


def _candidate(key, title, kind="faq"):
    return SimpleNamespace(
        key=key,
        retrieval_source=SimpleNamespace(id=key, title=title, kind=kind),
        distance=0.2,
        retrieval_path="dense",
    )


def _observe(
    before,
    after,
    *,
    text="",
    selected=None,
    visible=None,
    hook=True,
    active=True,
    case=None,
):
    return capture_scope_result(
        before=before,
        after=after,
        selected_ids=selected or [],
        sources=visible or [],
        context_text=text,
        diagnostics={"active": active, "strict_abstention": active} if hook else {},
        hook_available=hook,
        case=case
        or {"forbidden_titles": ["Office"], "forbidden_fact_keys": ["location"]},
        catalog={
            "facts": {"location": {"quote": "Office is in building X, room 117."}}
        },
    )


def _guard(**kwargs):
    return {
        "id": "public",
        "family": "scope",
        "expectation": "scope_guard_required",
        **kwargs,
    }


def test_passive_pool_capture_uses_postfilter_and_keeps_no_context_body():
    forbidden = _candidate("office", "Office")
    document = _candidate("doc", "Public process", "document")
    row = _observe(
        [forbidden, document], [document], text="public content", selected=["doc"]
    )
    assert [candidate["key"] for candidate in row["before_candidates"]] == [
        "office",
        "doc",
    ]
    assert [candidate["key"] for candidate in row["after_candidates"]] == ["doc"]
    assert row["excluded_keys"] == ["office"]
    assert "public content" not in str(row)
    assert row["forbidden_fact_hits"] == []
    validate_scope_observation(row)


def test_whole_result_detects_forbidden_fact_in_other_document():
    document = _candidate("doc", "Public process", "document")
    row = _observe(
        [document],
        [document],
        text="Document repeats: Office is in building X, room 117.",
        selected=["doc"],
    )
    baseline = _observe([document], [document], hook=False)
    assert row["forbidden_fact_hits"] == ["location"]
    assert compare_scope_case(_guard(), baseline, row)["failures"] == [
        "forbidden_fact_hits"
    ]


def test_legal_article_117_does_not_count_as_office_fact():
    row = _observe([], [], text="See legal article 117.")
    assert row["forbidden_fact_hits"] == []


def test_baseline_without_hook_records_identity_and_unknown_diagnostics():
    office = _candidate("office", "Office")
    baseline = _observe(
        [office],
        [office],
        hook=False,
        selected=["office"],
        visible=[office.retrieval_source],
    )
    candidate = _observe([office], [])
    result = compare_scope_case(_guard(), baseline, candidate)
    assert result["pass"]
    assert result["baseline_forbidden"]["forbidden_selected_keys"] == ["office"]


@pytest.mark.parametrize("stage", ["pool", "selected", "visible", "fact"])
def test_guard_rejects_each_surviving_forbidden_stage(stage):
    office = _candidate("office", "Office")
    baseline = _observe([office], [office], hook=False)
    row = _observe(
        [office],
        [office] if stage == "pool" else [],
        selected=["office"] if stage == "selected" else [],
        visible=[office.retrieval_source] if stage == "visible" else [],
        text="Office is in building X, room 117." if stage == "fact" else "",
    )
    assert not compare_scope_case(_guard(), baseline, row)["pass"]


def test_guard_requires_actual_candidate_hook_and_strict_diagnostics():
    baseline = _observe([], [], hook=False)
    assert not compare_scope_case(_guard(), baseline, baseline)["pass"]
    row = _observe([], [], active=False)
    assert compare_scope_case(_guard(), baseline, row)["failures"] == [
        "required_constraint_not_active",
        "required_strict_abstention_not_enabled",
    ]


def test_allowed_other_source_preserved_at_baseline_observed_stages():
    other = _candidate("other", "Archive")
    baseline = _observe(
        [other],
        [other],
        hook=False,
        selected=["other"],
        visible=[other.retrieval_source],
    )
    dropped = _observe([other], [])
    case = _guard(allowed_title="Archive", preserve_allowed_evidence=True)
    assert compare_scope_case(case, baseline, dropped)["failures"] == [
        "allowed_other_removed_by_filter",
        "allowed_other_candidate_regressed",
        "allowed_other_context_regressed",
        "allowed_other_visible_regressed",
    ]
    kept = _observe(
        [other], [other], selected=["other"], visible=[other.retrieval_source]
    )
    assert compare_scope_case(case, baseline, kept)["pass"]


def test_ordinary_control_checks_identity_not_invented_source_accuracy():
    doc = _candidate("doc", "Unknown process", "document")
    baseline = _observe([doc], [doc], hook=False, text="baseline", selected=["doc"])
    row = _observe([doc], [doc], active=False, text="baseline", selected=["doc"])
    case = {
        "id": "ordinary",
        "family": "ordinary",
        "expectation": "preserve_ordinary_path",
    }
    assert compare_scope_case(case, baseline, row)["pass"]
    changed = _observe([doc], [doc], active=False, text="changed", selected=["doc"])
    assert compare_scope_case(case, baseline, changed)["failures"] == [
        "ordinary_context_sha256_changed"
    ]


def test_missing_fact_controls_do_not_gain_content_credit():
    office = _candidate("office", "Computer Center")
    baseline = _observe([office], [office], hook=False)
    row = _observe([office], [office], active=False)
    case = {
        "id": "missing",
        "family": "missing",
        "expectation": "fact_missing_control",
        "expected_title": "Computer Center",
        "answer_fact_available": False,
    }
    result = compare_scope_case(case, baseline, row)
    assert result["pass"]
    assert result["content_pass"] is None


@pytest.mark.parametrize(
    "mutation", ["missing", "excluded", "duplicate", "baseline_diagnostics"]
)
def test_incomplete_or_fabricated_ledgers_fail_closed(mutation):
    office = _candidate("office", "Office")
    row = deepcopy(_observe([office], [], hook=True))
    if mutation == "missing":
        del row["forbidden_fact_hits"]
    elif mutation == "excluded":
        row["excluded_keys"] = []
    elif mutation == "duplicate":
        row["before_candidates"] *= 2
    else:
        row["hook_available"] = False
    with pytest.raises(
        ValueError, match=r"Incomplete|Inconsistent|Invalid|Unavailable"
    ):
        validate_scope_observation(row)


def test_passive_filter_capture_rejects_replaced_objects():
    with pytest.raises(ValueError, match="replaced"):
        _observe([_candidate("office", "Office")], [_candidate("office", "Office")])


def test_baseline_without_hook_cannot_claim_exclusion():
    with pytest.raises(ValueError, match="without hook"):
        _observe([_candidate("office", "Office")], [], hook=False)
