"""Strict scope pair gate plus a separate unchanged P1 preservation pair."""

# ruff: noqa: TRY301 -- CLI converts validity failures to bounded status JSON

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.eval.faq_evidence_scope_cases import (
    FACTS,
    FIXTURE,
    P1_FIXTURE_SHA256,
    load_fact_catalog,
    load_scope_cases,
)
from tests.eval.faq_evidence_scope_checks import compare_scope_case
from tests.eval.robust_faq_compare import (
    PAIRED_MANIFEST_KEYS,
    compare,
    validate_artifact,
)
from tests.eval.robust_faq_replay import digest, load_cases


def compare_scope(
    baseline: dict,
    candidate: dict,
    cases: list[dict],
    *,
    fixture_hash: str,
    catalog_hash: str,
) -> dict:
    left = validate_artifact(baseline, cases)
    right = validate_artifact(candidate, cases)
    for artifact in (baseline, candidate):
        if not isinstance(artifact.get("gpu_http_failures"), dict) or any(
            artifact["gpu_http_failures"].values()
        ):
            raise ValueError("Missing/nonzero GPU failure ledger")
        if artifact["manifest"].get("scope_catalog_sha256") != catalog_hash:
            raise ValueError("Scope catalog hash mismatch")
        if artifact["manifest"].get("fixture_sha256") != fixture_hash:
            raise ValueError("Scope fixture hash mismatch")
    for key in (*PAIRED_MANIFEST_KEYS, "evaluation_source_sha256"):
        if key not in baseline["manifest"] or baseline["manifest"][key] != candidate[
            "manifest"
        ].get(key):
            raise ValueError(f"Incomparable manifest field: {key}")
    if (
        not isinstance(baseline["manifest"]["evaluation_source_sha256"], dict)
        or not baseline["manifest"]["evaluation_source_sha256"]
    ):
        raise ValueError("Missing evaluation source fingerprints")
    rows = []
    for case in cases:
        before, after = left[case["id"]], right[case["id"]]
        if before.get("expected_id") != after.get("expected_id"):
            raise ValueError("Paired anchor IDs differ")
        for row in (before, after):
            observation = row["scope_observation"]
            forbidden = set(case.get("forbidden_titles", []))
            selected = set(row["selected_ids"])
            expected_flags = {
                "forbidden_candidate_keys": [
                    item["key"]
                    for item in observation["after_candidates"]
                    if item["kind"] == "faq" and item["title"] in forbidden
                ],
                "forbidden_selected_keys": [
                    item["key"]
                    for item in observation["before_candidates"]
                    if item["kind"] == "faq"
                    and item["title"] in forbidden
                    and item["key"] in selected
                ],
                "forbidden_visible_ids": [
                    item["id"]
                    for item in row["visible_sources"]
                    if item["kind"] == "faq" and item["title"] in forbidden
                ],
            }
            if any(
                observation[field] != value for field, value in expected_flags.items()
            ) or not set(observation["forbidden_fact_hits"]) <= set(
                case.get("forbidden_fact_keys", [])
            ):
                raise ValueError("Forbidden evidence ledger mismatch")
            if observation["hook_available"]:
                diagnostics = observation["diagnostics"]
                fallback = row["fallback"]
                if (
                    diagnostics["active"] != fallback["faq_evidence_constraint_applied"]
                    or diagnostics["reason"]
                    != fallback["faq_evidence_constraint_reason"]
                    or diagnostics["excluded_candidate_count"]
                    != len(observation["excluded_keys"])
                    or diagnostics["excluded_candidate_count"]
                    != fallback["faq_evidence_excluded_candidate_count"]
                    or diagnostics["constrained_abstained"]
                    != fallback["faq_evidence_constrained_abstained"]
                ):
                    raise ValueError("Constraint diagnostics ledger mismatch")
            if any(
                row[field] != observation[scope_field]
                for field, scope_field in (
                    ("candidates", "after_candidates"),
                    ("selected_ids", "selected_ids"),
                    ("visible_sources", "visible_sources"),
                    ("context_sha256", "context_sha256"),
                )
            ):
                raise ValueError("Scope/result observation mismatch")
        result = compare_scope_case(
            case, before["scope_observation"], after["scope_observation"]
        )
        if before.get("content_pass") is True and after.get("content_pass") is not True:
            result["failures"].append("positive_content_regressed")
        result["pass"] = not result["failures"]
        rows.append(result)
    families: dict[str, dict[str, int]] = {}
    for row in rows:
        group = families.setdefault(row["family"], {"cases": 0, "failed": 0})
        group["cases"] += 1
        group["failed"] += not row["pass"]
    return {
        "valid": True,
        "scope_gate_pass": all(row["pass"] for row in rows),
        "cases": len(rows),
        "families": families,
        "results": rows,
        "baseline_source_manifest_sha256": digest(
            baseline["manifest"]["source_sha256"]
        ),
        "candidate_source_manifest_sha256": digest(
            candidate["manifest"]["source_sha256"]
        ),
        "scope": "authored retrieval diagnostics only; no generation/answer accuracy; independent link enrichment not observed",
    }


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    for name in (
        "baseline",
        "candidate",
        "preservation-baseline",
        "preservation-candidate",
    ):
        cli.add_argument(f"--{name}", required=True)
    cli.add_argument("--fixture", default=str(FIXTURE))
    cli.add_argument("--scope-catalog", default=str(FACTS))
    ns = cli.parse_args()
    try:
        fixture, catalog = Path(ns.fixture), Path(ns.scope_catalog)
        report = compare_scope(
            _read(ns.baseline),
            _read(ns.candidate),
            load_scope_cases(fixture, load_fact_catalog(catalog)),
            fixture_hash=_hash(fixture),
            catalog_hash=_hash(catalog),
        )
        p1 = FIXTURE.with_name("robust_faq_golden.jsonl")
        if _hash(p1) != P1_FIXTURE_SHA256:
            raise ValueError("Original P1 fixture changed")
        before, after = (
            _read(ns.preservation_baseline),
            _read(ns.preservation_candidate),
        )
        preservation = compare(
            before, after, load_cases(p1), fixture_hash=P1_FIXTURE_SHA256
        )
        for scope_run, p1_run in (
            (_read(ns.baseline), before),
            (_read(ns.candidate), after),
        ):
            for key in (set(PAIRED_MANIFEST_KEYS) | {"evaluation_source_sha256"}) - {
                "fixture_sha256"
            }:
                if scope_run["manifest"][key] != p1_run["manifest"][key]:
                    raise ValueError(f"Scope/preservation runtime differs: {key}")
            if (
                scope_run["manifest"]["source_sha256"]
                != p1_run["manifest"]["source_sha256"]
            ):
                raise ValueError("Scope/preservation application differs")
            if not isinstance(p1_run.get("gpu_http_failures"), dict) or any(
                p1_run["gpu_http_failures"].values()
            ):
                raise ValueError("Preservation GPU failures/missing ledger")
        report["preservation"] = preservation
        report["release_gate_pass"] = (
            report["scope_gate_pass"] and preservation["release_gate_pass"]
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"valid": False, "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["release_gate_pass"] else 1


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
