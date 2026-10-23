"""Pure comparison of saved public production-path retrieval observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tests.eval.robust_faq_replay import digest, load_cases, summarize

PAIRED_MANIFEST_KEYS = (
    "fixture_sha256",
    "gpu_manifest_sha256",
    "corpus",
    "embedding_model",
    "bge_spec",
    "embedding_dimensions",
    "dense_cutoff",
    "context_floor",
    "source_floor",
    "initial_k",
    "top_k",
    "credentials_supplied",
    "python",
    "dependencies",
    "transport",
)


def validate_read_accounting(failures) -> None:
    if (
        not isinstance(failures, dict)
        or "total" not in failures
        or not isinstance(failures.get("by_error_type"), dict)
    ):
        raise ValueError("Database read accounting missing")
    if failures["total"] or any(failures["by_error_type"].values()):
        raise ValueError("Recorded database read failures invalidate replay")


def validate_artifact(artifact: dict, cases: list[dict]) -> dict[str, dict]:
    validate_read_accounting(artifact.get("db_read_failures"))
    if (
        artifact.get("schema_version") != "robust-faq-replay-v1"
        or artifact.get("status") != "completed"
    ):
        raise ValueError("Run is not a completed production replay")
    if artifact.get("preflight", {}).get("valid") is not True:
        raise ValueError("Corpus preflight incomplete")
    if not artifact.get("manifest", {}).get("source_sha256"):
        raise ValueError("Source fingerprints missing")
    rows = artifact.get("results", [])
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {case["id"] for case in cases}:
        raise ValueError("Duplicate, missing or extra case records")
    for case in cases:
        row = by_id[case["id"]]
        if row.get("execution_status") != "completed":
            raise ValueError("Execution error in completed artifact")
        validate_read_accounting(row.get("db_read_failures"))
        if any(row.get(key) != case[key] for key in ("family", "split", "expectation")):
            raise ValueError("Case metadata differs from fixture")
        if row.get("answer_fact_available") != case.get("answer_fact_available"):
            raise ValueError("Fact availability metadata differs from fixture")
        if (
            row.get("target_visible_hit") is True
            and row.get("target_context_hit") is not True
        ):
            raise ValueError("Visible target absent from selected context")
        if (
            row.get("content_pass") is True
            and row.get("target_context_hit") is not True
        ):
            raise ValueError("Content pass without selected target")
        if (
            type(row.get("n_candidates")) is not int
            or not 0 <= row["n_candidates"] <= 125
        ):
            raise ValueError("Invalid candidate count")
        if len(row.get("candidates", [])) != row["n_candidates"]:
            raise ValueError("Candidate observations incomplete")
        if not isinstance(row.get("local_recovery_activated"), bool):
            raise TypeError("Missing recovery observation")
        pool = row["candidates"]
        keys = {item["key"] for item in pool}
        selected = set(row.get("selected_ids", []))
        if len(keys) != len(pool) or not selected <= keys or len(selected) > 10:
            raise ValueError("Candidate/selection identities inconsistent")
        visible = {
            (item["kind"], item["id"]) for item in row.get("visible_sources", [])
        }
        selected_sources = {
            (item["kind"], item["id"]) for item in pool if item["key"] in selected
        }
        if not visible <= selected_sources:
            raise ValueError("Visible source is not selected")
        if case.get("expected_title"):
            if row.get("expected_title") != case["expected_title"] or not row.get(
                "expected_id"
            ):
                raise ValueError("Expected anchor missing")
            if any(
                not isinstance(row.get(key), bool)
                for key in (
                    "target_candidate_hit",
                    "target_context_hit",
                    "target_visible_hit",
                )
            ):
                raise ValueError("Missing positive stage observations")
            if case.get("answer_fact_available") is False:
                if row.get("content_pass") is not None:
                    raise ValueError("Missing-fact control cannot pass a fact check")
            elif not isinstance(row.get("content_pass"), bool):
                raise TypeError("Missing content observation")
            anchor = row["expected_id"]
            target = next(
                (
                    item
                    for item in pool
                    if item["kind"] == "faq" and item["id"] == anchor
                ),
                None,
            )
            if row["target_candidate_hit"] != bool(target):
                raise ValueError("Target candidate membership inconsistent")
            if row["target_context_hit"] and (
                not target or target["key"] not in selected
            ):
                raise ValueError("Target context selection inconsistent")
            if row["target_visible_hit"] != (("faq", anchor) in visible):
                raise ValueError("Target visibility inconsistent")
        else:
            forbidden = case.get("forbidden_recovery_title")
            leakage = any(
                item["key"] in selected and item["title"] == forbidden for item in pool
            )
            if row.get("forbidden_context_hit") != leakage:
                raise ValueError("Negative leakage observation inconsistent")
    return by_id


def compare(
    baseline: dict, candidate: dict, cases: list[dict], *, fixture_hash: str
) -> dict:
    left = validate_artifact(baseline, cases)
    right = validate_artifact(candidate, cases)
    for key in PAIRED_MANIFEST_KEYS:
        if (
            key not in baseline["manifest"]
            or key not in candidate["manifest"]
            or baseline["manifest"][key] != candidate["manifest"][key]
        ):
            raise ValueError(f"Incomparable manifest field: {key}")
    if baseline["manifest"]["fixture_sha256"] != fixture_hash:
        raise ValueError("Fixture hash mismatch")
    fixed, regressions, unchanged_misses, required_failures, negative_failures = (
        [],
        [],
        [],
        [],
        [],
    )
    family_results: dict[str, dict] = {}
    for case in cases:
        before, after = left[case["id"]], right[case["id"]]
        if before.get("expected_id") != after.get("expected_id"):
            raise ValueError("Paired anchor IDs differ")
        positive = case["expectation"] != "no_local_entity_recovery"
        family = family_results.setdefault(
            case["family"],
            {
                "cases": 0,
                "fixed": 0,
                "regressions": 0,
                "required_failures": 0,
                "negative_failures": 0,
            },
        )
        family["cases"] += 1
        if positive:
            if not before["target_context_hit"] and after["target_context_hit"]:
                fixed.append(case["id"])
                family["fixed"] += 1
            if not before["target_context_hit"] and not after["target_context_hit"]:
                unchanged_misses.append(case["id"])
            if any(
                before[key] and not after[key]
                for key in (
                    "target_candidate_hit",
                    "target_context_hit",
                    "content_pass",
                    "target_visible_hit",
                )
            ):
                regressions.append(case["id"])
                family["regressions"] += 1
            if case["expectation"] == "fix_required" and not all(
                after[key]
                for key in (
                    "target_candidate_hit",
                    "target_context_hit",
                    "content_pass",
                    "target_visible_hit",
                )
            ):
                required_failures.append(case["id"])
                family["required_failures"] += 1
        elif after["local_recovery_activated"] or (
            after["forbidden_context_hit"] and not before["forbidden_context_hit"]
        ):
            negative_failures.append(case["id"])
            family["negative_failures"] += 1
    return {
        "valid": True,
        "release_gate_pass": not (
            regressions or required_failures or negative_failures
        ),
        "baseline": summarize(list(left.values())),
        "candidate": summarize(list(right.values())),
        "fixed": fixed,
        "regressions": regressions,
        "unchanged_misses": unchanged_misses,
        "required_failures": required_failures,
        "negative_failures": negative_failures,
        "families": family_results,
        "baseline_source_manifest_sha256": digest(
            baseline["manifest"].get("source_sha256")
        ),
        "candidate_source_manifest_sha256": digest(
            candidate["manifest"].get("source_sha256")
        ),
        "scope": "retrieval comparison only; no answer accuracy claim; designated holdouts are not blinded",
    }


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", required=True)
    cli.add_argument("--candidate", required=True)
    cli.add_argument(
        "--fixture", default=str(Path(__file__).with_name("robust_faq_golden.jsonl"))
    )
    ns = cli.parse_args()
    try:
        fixture = Path(ns.fixture)

        report = compare(
            json.loads(Path(ns.baseline).read_text(encoding="utf-8")),
            json.loads(Path(ns.candidate).read_text(encoding="utf-8")),
            load_cases(fixture),
            fixture_hash=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"valid": False, "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["release_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
