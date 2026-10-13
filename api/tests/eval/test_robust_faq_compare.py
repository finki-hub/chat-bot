from copy import deepcopy

import pytest

from tests.eval.robust_faq_compare import PAIRED_MANIFEST_KEYS, compare


def inputs():
    cases = [
        {
            "id": "positive",
            "family": "location",
            "split": "control",
            "expectation": "fix_required",
            "expected_title": "FAQ",
        },
        {
            "id": "negative",
            "family": "other_service",
            "split": "control",
            "expectation": "no_local_entity_recovery",
            "forbidden_recovery_title": "FAQ",
        },
    ]
    manifest: dict = dict.fromkeys(PAIRED_MANIFEST_KEYS, "same")
    manifest["fixture_sha256"] = "fixture"
    manifest["source_sha256"] = {"llms/context.py": "source"}
    rows: list[dict] = [
        {
            **case,
            "expected_id": "a" if case["id"] == "positive" else None,
            "execution_status": "completed",
            "effective_transform_mode": "raw",
            "n_candidates": 0,
            "candidates": [],
            "selected_ids": [],
            "visible_sources": [],
            "target_candidate_hit": False,
            "target_context_hit": False,
            "target_visible_hit": False,
            "content_pass": False,
            "local_recovery_activated": False,
            "forbidden_context_hit": False,
            "db_read_failures": {"total": 0, "by_error_type": {}},
        }
        for case in cases
    ]
    baseline: dict = {
        "schema_version": "robust-faq-replay-v1",
        "status": "completed",
        "manifest": manifest,
        "results": rows,
        "preflight": {"valid": True},
        "db_read_failures": {"total": 0, "by_error_type": {}},
    }
    candidate = deepcopy(baseline)
    candidate["results"][0].update(
        target_candidate_hit=True,
        target_context_hit=True,
        target_visible_hit=True,
        content_pass=True,
        n_candidates=1,
        candidates=[{"id": "a", "key": "Q:a", "kind": "faq", "title": "FAQ"}],
        selected_ids=["Q:a"],
        visible_sources=[{"id": "a", "kind": "faq", "title": "FAQ"}],
    )
    return baseline, candidate, cases


def test_fixed_target_and_negative_control_pass():
    before, after, cases = inputs()
    report = compare(before, after, cases, fixture_hash="fixture")
    assert report["release_gate_pass"] is True
    assert report["fixed"] == ["positive"]
    assert report["candidate"]["completed"] == 2


@pytest.mark.parametrize("key", PAIRED_MANIFEST_KEYS)
def test_manifest_changes_invalidate_pair(key):
    before, after, cases = inputs()
    after["manifest"][key] = "different"
    with pytest.raises(ValueError, match="Incomparable"):
        compare(before, after, cases, fixture_hash="fixture")


@pytest.mark.parametrize(
    "failure",
    ["missing", "duplicate", "error", "visible_without_context", "missing_candidates"],
)
def test_bad_records_invalidate_comparison(failure):
    before, after, cases = inputs()
    if failure == "missing":
        after["results"].pop()
    elif failure == "duplicate":
        after["results"].append(after["results"][0])
    elif failure == "error":
        after["results"][0]["execution_status"] = "error"
    elif failure == "visible_without_context":
        after["results"][0]["target_context_hit"] = False
    else:
        after["results"][0]["candidates"] = []
    with pytest.raises(ValueError, match=r"records|error|context|incomplete"):
        compare(before, after, cases, fixture_hash="fixture")


def test_negative_activation_blocks_release_even_without_visible_source():
    before, after, cases = inputs()
    after["results"][1]["local_recovery_activated"] = True
    report = compare(before, after, cases, fixture_hash="fixture")
    assert report["release_gate_pass"] is False
    assert report["negative_failures"] == ["negative"]


def test_existing_negative_leakage_is_reported_not_new_failure():
    before, after, cases = inputs()
    before["results"][1]["forbidden_context_hit"] = True
    after["results"][1]["forbidden_context_hit"] = True
    for artifact in (before, after):
        artifact["results"][1].update(
            n_candidates=1,
            candidates=[{"id": "a", "key": "Q:a", "kind": "faq", "title": "FAQ"}],
            selected_ids=["Q:a"],
        )
    report = compare(before, after, cases, fixture_hash="fixture")
    assert report["release_gate_pass"] is True
    assert (
        report["baseline"]["groups"]["no_local_entity_recovery"][
            "forbidden_context_hits"
        ]
        == 1
    )


def test_visibility_regression_blocks_release():
    before, after, cases = inputs()
    before["results"][0] = deepcopy(after["results"][0])
    after["results"][0]["target_visible_hit"] = False
    after["results"][0]["visible_sources"] = []
    report = compare(before, after, cases, fixture_hash="fixture")
    assert report["release_gate_pass"] is False
    assert report["regressions"] == ["positive"]


def test_required_recovery_without_visible_source_blocks_release():
    before, after, cases = inputs()
    after["results"][0]["target_visible_hit"] = False
    after["results"][0]["visible_sources"] = []
    report = compare(before, after, cases, fixture_hash="fixture")
    assert after["results"][0]["target_candidate_hit"] is True
    assert after["results"][0]["target_context_hit"] is True
    assert after["results"][0]["content_pass"] is True
    assert report["release_gate_pass"] is False
    assert report["required_failures"] == ["positive"]
    assert report["regressions"] == []


@pytest.mark.parametrize(
    "failure",
    [
        "preflight",
        "source",
        "candidate_hit",
        "selection",
        "visibility",
        "negative_leakage",
    ],
)
def test_stage_integrity_cannot_be_fabricated(failure):
    before, after, cases = inputs()
    if failure == "preflight":
        after["preflight"]["valid"] = False
    elif failure == "source":
        after["manifest"]["source_sha256"] = {}
    elif failure == "candidate_hit":
        after["results"][0]["target_candidate_hit"] = False
    elif failure == "selection":
        after["results"][0]["selected_ids"] = ["Q:not-in-pool"]
    elif failure == "visibility":
        after["results"][0]["visible_sources"] = []
    else:
        after["results"][1]["forbidden_context_hit"] = True
    with pytest.raises(ValueError, match=r"preflight|fingerprints|inconsistent"):
        compare(before, after, cases, fixture_hash="fixture")


def test_retrieved_title_is_not_credit_for_an_unavailable_fact():
    before, after, cases = inputs()
    cases[0].update(expectation="preserve_or_diagnose", answer_fact_available=False)
    for artifact in (before, after):
        artifact["results"][0].update(
            expectation="preserve_or_diagnose",
            answer_fact_available=False,
            content_pass=None,
        )
    report = compare(before, after, cases, fixture_hash="fixture")
    assert report["release_gate_pass"] is True
    assert report["candidate"]["missing_fact_controls"] == 1
    assert report["candidate"]["selected_faq_without_requested_fact"] == 1
    assert report["candidate"]["groups"]["preserve_or_diagnose"]["content_passes"] == 0
    after["results"][0]["content_pass"] = True
    with pytest.raises(ValueError, match="Missing-fact"):
        compare(before, after, cases, fixture_hash="fixture")


@pytest.mark.parametrize("location", ["artifact", "case"])
@pytest.mark.parametrize(
    "counts",
    [
        {"total": 1, "by_error_type": {"TimeoutError": 1}},
        {"total": 0, "by_error_type": {"PostgresError": 1}},
    ],
)
def test_completed_status_cannot_hide_recorded_database_faults(location, counts):
    before, after, cases = inputs()
    target = after if location == "artifact" else after["results"][0]
    target["db_read_failures"] = counts
    with pytest.raises(ValueError, match="database read failures"):
        compare(before, after, cases, fixture_hash="fixture")


def test_zero_database_faults_are_not_a_paired_manifest_field():
    before, after, cases = inputs()
    before["db_read_failures"] = {"total": 0, "by_error_type": {}}
    after["db_read_failures"] = {"total": 0, "by_error_type": {"TimeoutError": 0}}
    assert "db_read_failures" not in PAIRED_MANIFEST_KEYS
    assert (
        compare(before, after, cases, fixture_hash="fixture")["release_gate_pass"]
        is True
    )


@pytest.mark.parametrize("location", ["artifact", "case"])
def test_old_completed_artifacts_without_read_accounting_require_rerun(location):
    before, after, cases = inputs()
    target = after if location == "artifact" else after["results"][0]
    target.pop("db_read_failures")
    with pytest.raises(ValueError, match="accounting missing"):
        compare(before, after, cases, fixture_hash="fixture")
