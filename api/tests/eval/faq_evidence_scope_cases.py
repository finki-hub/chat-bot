"""Public declarative evaluation inputs; never classifies a production query."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).with_name("faq_evidence_scope_golden.jsonl")
FACTS = Path(__file__).with_name("faq_evidence_scope_facts.json")
EXPECTATIONS = {
    "scope_guard_required",
    "preserve_local_evidence",
    "preserve_ordinary_path",
    "fact_missing_control",
}
P1_FIXTURE_SHA256 = "1ffcb8b32ca97f327bb1d483ddb088f20ed3da2afce0cf7334961aec8152e1b5"


def load_fact_catalog(path: Path = FACTS) -> dict:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog["schema_version"] != "public-office-evidence-v1":
        raise ValueError("Unknown public evidence catalog schema")
    for fact in catalog["facts"].values():
        if fact["source_title"] not in catalog["sources"] or not fact["quote"]:
            raise ValueError("Missing public fact source/quote")
    for group in catalog["groups"].values():
        if any(title not in catalog["sources"] for title in group["titles"]):
            raise ValueError("Unknown group source")
        if any(key not in catalog["facts"] for key in group["fact_keys"]):
            raise ValueError("Unknown group fact")
    return catalog


def load_scope_cases(path: Path = FIXTURE, catalog: dict | None = None) -> list[dict]:
    catalog = catalog or load_fact_catalog()
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if (
        not cases
        or len(cases) > 40
        or len({case["id"] for case in cases}) != len(cases)
    ):
        raise ValueError("Scope fixture requires 1-40 unique public cases")
    for case in cases:
        if case["expectation"] not in EXPECTATIONS:
            raise ValueError("Unknown scope expectation")
        if not all(
            isinstance(case.get(key), str) and case[key]
            for key in ("id", "query", "family", "split", "scope_requirement")
        ):
            raise ValueError("Missing public case identity")
        group = catalog["groups"].get(case.get("forbidden_group"))
        if case["expectation"] == "scope_guard_required" and group is None:
            raise ValueError("Missing forbidden evidence group")
        if case["expectation"] == "preserve_ordinary_path" and (
            case["scope_requirement"] != "none" or case.get("expected_title") or group
        ):
            raise ValueError("Ordinary controls cannot manufacture scope/source gold")
        if case["expectation"] in {
            "preserve_local_evidence",
            "fact_missing_control",
        } and not case.get("expected_title"):
            raise ValueError("Missing declared public anchor")
        if case["expectation"] == "fact_missing_control" and (
            case.get("answer_fact_available") is not False or case.get("context_tokens")
        ):
            raise ValueError("Missing-fact control cannot claim content credit")
        case["forbidden_titles"] = list(group["titles"]) if group else []
        case["forbidden_fact_keys"] = list(group["fact_keys"]) if group else []
        if case.get("allowed_title") in case["forbidden_titles"]:
            raise ValueError("Allowed and forbidden public anchors conflict")
    return cases


def preflight_scope_cases(
    cases: list[dict], catalog: dict, manifest: dict, bodies: list[dict] | None = None
) -> dict:
    names: dict[str, list[str]] = {}
    for row in manifest["titles"]:
        names.setdefault(row["name"], []).append(str(row["id"]))
    wanted = set(catalog["sources"])
    wanted.update(
        case["expected_title"] for case in cases if case.get("expected_title")
    )
    wanted.update(case["allowed_title"] for case in cases if case.get("allowed_title"))
    errors = [
        {"source_title": name, "reason": "missing_or_ambiguous_title"}
        for name in sorted(wanted)
        if len(names.get(name, [])) != 1
    ]
    content = {row["name"]: row["content"] for row in bodies or []}
    pending = []
    for title, source in catalog["sources"].items():
        if (
            title in content
            and hashlib.md5(content[title].encode(), usedforsecurity=False).hexdigest()
            != source["body_md5"]
        ):
            errors.append(
                {"source_title": title, "reason": "reviewed_public_body_changed"}
            )
    for key, fact in catalog["facts"].items():
        title = fact["source_title"]
        if title not in content:
            pending.append(key)
        elif fact["quote"] not in content[title]:
            errors.append({"fact_key": key, "reason": "public_quote_absent"})
    for case in cases:
        title = case.get("expected_title")
        if case.get("context_tokens") and title not in content:
            pending.append(case["id"])
        elif title in content and any(
            token not in content[title] for token in case.get("context_tokens", [])
        ):
            errors.append({"id": case["id"], "reason": "public_token_absent"})
    return {
        "valid": not errors and not pending,
        "errors": errors,
        "pending": pending,
        "resolved_titles": {
            name: names[name][0]
            for name in sorted(wanted)
            if len(names.get(name, [])) == 1
        },
        "cases": len(cases),
        "families": len({case["family"] for case in cases}),
        "scope": "quote occurrence/public corpus consistency, not entailment or answer accuracy",
    }
