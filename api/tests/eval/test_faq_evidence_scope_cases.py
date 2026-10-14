import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from tests.eval import faq_evidence_scope_cases as cases_module


def test_public_scope_fixture_is_bounded_and_p1_fixture_unchanged():
    cases = cases_module.load_scope_cases()
    assert len(cases) == 38
    assert Counter(case["expectation"] for case in cases) == {
        "scope_guard_required": 14,
        "preserve_local_evidence": 12,
        "preserve_ordinary_path": 10,
        "fact_missing_control": 2,
    }
    p1 = cases_module.FIXTURE.with_name("robust_faq_golden.jsonl")
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == cases_module.P1_FIXTURE_SHA256


def test_known_regression_queries_match_public_p1_definitions_exactly():
    original = {
        row["id"]: row
        for row in [
            json.loads(line)
            for line in cases_module.FIXTURE.with_name("robust_faq_golden.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        ]
    }
    for case in cases_module.load_scope_cases():
        if case.get("p1_case_id"):
            assert case["query"] == original[case["p1_case_id"]]["query"]


def test_ordinary_controls_have_no_fabricated_source_gold():
    for case in cases_module.load_scope_cases():
        if case["expectation"] == "preserve_ordinary_path":
            assert case["scope_requirement"] == "none"
            assert case["forbidden_titles"] == []
            assert case.get("expected_title") is None


def test_other_office_is_not_declared_forbidden_by_entity_exclusion():
    for case in cases_module.load_scope_cases():
        if case.get("allowed_title"):
            assert case["allowed_title"] not in case["forbidden_titles"]


def test_no_bare_117_as_forbidden_fact_pattern():
    catalog = cases_module.load_fact_catalog()
    assert all(fact["quote"] != "117" for fact in catalog["facts"].values())
    assert "117" in catalog["facts"]["service_location"]["quote"]
    assert len(catalog["facts"]["service_location"]["quote"]) > 50


def test_preflight_does_not_claim_missing_current_bodies_were_checked():
    catalog = cases_module.load_fact_catalog()
    manifest = {
        "titles": [
            {"name": name, "id": str(index)}
            for index, name in enumerate(catalog["sources"])
        ]
    }
    result = cases_module.preflight_scope_cases(
        cases_module.load_scope_cases(), catalog, manifest
    )
    assert result["valid"] is False
    assert result["errors"] == []
    assert "general_contact_email" in result["pending"]
    assert len(result["resolved_titles"]) == 5


def test_preflight_rejects_changed_reviewed_body_and_missing_quote():
    catalog = {
        "sources": {
            "Public FAQ": {
                "body_md5": hashlib.md5(
                    b"known quote", usedforsecurity=False
                ).hexdigest()
            }
        },
        "facts": {
            "public_fact": {"source_title": "Public FAQ", "quote": "known quote"}
        },
    }
    manifest = {"titles": [{"name": "Public FAQ", "id": "public"}]}
    result = cases_module.preflight_scope_cases(
        [], catalog, manifest, [{"name": "Public FAQ", "content": "changed"}]
    )
    assert result["valid"] is False
    assert {error["reason"] for error in result["errors"]} == {
        "reviewed_public_body_changed",
        "public_quote_absent",
    }


@pytest.mark.parametrize(
    "mutation",
    ["unknown_group", "ordinary_gold", "conflicting_anchor", "missing_fact_credit"],
)
def test_invalid_declared_gold_is_rejected(tmp_path: Path, mutation):
    catalog = cases_module.load_fact_catalog()
    row = deepcopy(cases_module.load_scope_cases()[0])
    if mutation == "unknown_group":
        row["forbidden_group"] = "missing"
    elif mutation == "ordinary_gold":
        row.update(
            expectation="preserve_ordinary_path",
            scope_requirement="none",
            expected_title="Студентска служба",
        )
    elif mutation == "conflicting_anchor":
        row["allowed_title"] = "Студентска служба"
    else:
        row.update(
            expectation="fact_missing_control",
            expected_title="Компјутерски центар",
            answer_fact_available=False,
            context_tokens=["invented location"],
        )
    path = tmp_path / "public.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match=r"group|gold|conflict|credit"):
        cases_module.load_scope_cases(path, catalog)
