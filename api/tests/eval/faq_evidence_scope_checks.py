"""Content-free scope observations and pure paired assertions, not query routing.

The runner must supply actual objects from production filter/result boundaries.
This module neither calls a model nor parses an institution or negated entity.
"""

from __future__ import annotations

from tests.eval.robust_faq_replay import MAX_CANDIDATES, candidate_record, digest


def capture_scope_result(
    *,
    before: list,
    after: list,
    selected_ids: list[str],
    sources: list,
    context_text: str,
    diagnostics: dict,
    hook_available: bool,
    case: dict,
    catalog: dict,
) -> dict:
    """Serialize real pools and quote-hit flags, never corpus text or score inputs."""
    before_keys = [candidate.key for candidate in before]
    after_keys = [candidate.key for candidate in after]
    if (
        max(len(before), len(after)) > MAX_CANDIDATES
        or len(set(before_keys)) != len(before_keys)
        or len(set(after_keys)) != len(after_keys)
        or not set(after_keys).issubset(before_keys)
    ):
        raise ValueError("Invalid actual scope pool observation")
    original = {candidate.key: candidate for candidate in before}
    if any(candidate is not original[candidate.key] for candidate in after):
        raise ValueError("Scope filter replaced candidate objects")
    if not hook_available and before_keys != after_keys:
        raise ValueError("Baseline without hook cannot claim filtering")
    selected = set(selected_ids)
    forbidden = set(case.get("forbidden_titles", []))
    visible = [
        {"id": source.id, "kind": source.kind, "title": source.title}
        for source in sources
    ]
    # Inspect the whole final context: forbidden quotes may recur in documents.
    fact_hits = [
        key
        for key in case.get("forbidden_fact_keys", [])
        if catalog["facts"][key]["quote"] in context_text
    ]
    selected_forbidden = [
        candidate.key
        for candidate in before
        if candidate.key in selected
        and candidate.retrieval_source.kind == "faq"
        and candidate.retrieval_source.title in forbidden
    ]
    return {
        "schema_version": "scope-observation-v1",
        "hook_available": hook_available,
        "diagnostics": diagnostics,
        "before_candidates": [candidate_record(candidate) for candidate in before],
        "after_candidates": [candidate_record(candidate) for candidate in after],
        "excluded_keys": [key for key in before_keys if key not in set(after_keys)],
        "selected_ids": list(selected_ids),
        "visible_sources": visible,
        "context_sha256": digest(context_text),
        "forbidden_candidate_keys": [
            candidate.key
            for candidate in after
            if candidate.retrieval_source.kind == "faq"
            and candidate.retrieval_source.title in forbidden
        ],
        "forbidden_selected_keys": selected_forbidden,
        "forbidden_visible_ids": [
            source["id"]
            for source in visible
            if source["kind"] == "faq" and source["title"] in forbidden
        ],
        "forbidden_fact_hits": fact_hits,
    }


def validate_scope_observation(observation: dict) -> None:
    """Fail closed on incomplete bookkeeping; no unavailable-as-zero defaults."""
    required = {
        "schema_version",
        "hook_available",
        "diagnostics",
        "before_candidates",
        "after_candidates",
        "excluded_keys",
        "selected_ids",
        "visible_sources",
        "context_sha256",
        "forbidden_candidate_keys",
        "forbidden_selected_keys",
        "forbidden_visible_ids",
        "forbidden_fact_hits",
    }
    if (
        not required.issubset(observation)
        or observation["schema_version"] != "scope-observation-v1"
    ):
        raise ValueError("Incomplete scope observation")
    if not isinstance(observation["hook_available"], bool) or not isinstance(
        observation["diagnostics"], dict
    ):
        raise TypeError("Invalid scope hook diagnostics")
    for field in required - {
        "schema_version",
        "hook_available",
        "diagnostics",
        "context_sha256",
    }:
        if not isinstance(observation[field], list):
            raise TypeError("Invalid scope observation list")
    if (
        not isinstance(observation["context_sha256"], str)
        or len(observation["context_sha256"]) != 64
    ):
        raise ValueError("Missing final context fingerprint")
    before = observation["before_candidates"]
    after = observation["after_candidates"]
    before_keys = [candidate["key"] for candidate in before]
    after_keys = [candidate["key"] for candidate in after]
    if (
        max(len(before), len(after)) > MAX_CANDIDATES
        or len(set(before_keys)) != len(before_keys)
        or len(set(after_keys)) != len(after_keys)
    ):
        raise ValueError("Invalid scope pool keys/budget")
    if not set(after_keys).issubset(before_keys) or observation["excluded_keys"] != [
        key for key in before_keys if key not in set(after_keys)
    ]:
        raise ValueError("Inconsistent actual scope exclusion ledger")
    if not observation["hook_available"] and (
        before != after or observation["diagnostics"]
    ):
        raise ValueError("Unavailable baseline hook has fabricated diagnostics/filter")


def _title_hit(observation: dict, title: str, stage: str) -> bool:
    if stage == "visible":
        return any(
            source["kind"] == "faq" and source["title"] == title
            for source in observation["visible_sources"]
        )
    selected = set(observation["selected_ids"])
    return any(
        candidate["kind"] == "faq"
        and candidate["title"] == title
        and (stage == "candidate" or candidate["key"] in selected)
        for candidate in observation["after_candidates"]
    )


def compare_scope_case(case: dict, baseline: dict, candidate: dict) -> dict:
    """Layer on a separately validated run pair; this is not a validity gate."""
    validate_scope_observation(baseline)
    validate_scope_observation(candidate)
    errors = []
    expectation = case["expectation"]
    if not candidate["hook_available"]:
        errors.append("candidate_filter_hook_unavailable")
    if expectation == "scope_guard_required":
        if candidate["diagnostics"].get("active") is not True:
            errors.append("required_constraint_not_active")
        if candidate["diagnostics"].get("strict_abstention") is not True:
            errors.append("required_strict_abstention_not_enabled")
        errors.extend(
            field
            for field in (
                "forbidden_candidate_keys",
                "forbidden_selected_keys",
                "forbidden_visible_ids",
                "forbidden_fact_hits",
            )
            if candidate[field]
        )
        if case.get("preserve_allowed_evidence"):
            allowed = case["allowed_title"]
            if any(
                item["kind"] == "faq" and item["title"] == allowed
                for item in candidate["before_candidates"]
            ) and not _title_hit(candidate, allowed, "candidate"):
                errors.append("allowed_other_removed_by_filter")
            errors.extend(
                f"allowed_other_{stage}_regressed"
                for stage in ("candidate", "context", "visible")
                if _title_hit(baseline, case["allowed_title"], stage)
                and not _title_hit(candidate, case["allowed_title"], stage)
            )
    elif expectation == "preserve_ordinary_path":
        if candidate["diagnostics"].get("active") is not False:
            errors.append("ordinary_control_constrained_or_unknown")
        errors.extend(
            f"ordinary_{field}_changed"
            for field in (
                "before_candidates",
                "after_candidates",
                "selected_ids",
                "visible_sources",
                "context_sha256",
            )
            if baseline[field] != candidate[field]
        )
    elif expectation in {"preserve_local_evidence", "fact_missing_control"}:
        errors.extend(
            f"local_{stage}_regressed"
            for stage in ("candidate", "context", "visible")
            if _title_hit(baseline, case["expected_title"], stage)
            and not _title_hit(candidate, case["expected_title"], stage)
        )
    else:
        raise ValueError("Unknown scope expectation")
    return {
        "id": case["id"],
        "family": case["family"],
        "expectation": expectation,
        "pass": not errors,
        "failures": errors,
        "baseline_forbidden": {
            field: baseline[field]
            for field in (
                "forbidden_candidate_keys",
                "forbidden_selected_keys",
                "forbidden_visible_ids",
                "forbidden_fact_hits",
            )
        },
        "candidate_forbidden": {
            field: candidate[field]
            for field in (
                "forbidden_candidate_keys",
                "forbidden_selected_keys",
                "forbidden_visible_ids",
                "forbidden_fact_hits",
            )
        },
        "content_pass": None,
        "content_check_performed": False,
        "scope": "retrieval-stage assertions only; pair validity/content token gate checked separately; no answer accuracy",
    }
