"""Opt-in, sequential, retrieval-only observation of the production function.

Import is offline. CLI execution requires parent-authorized read-only DB/GPU access.
Never generates answers, runs migrations, exports corpus bodies, or logs credentials.
"""

# ruff: noqa: PLC0415, TRY301
# Application imports follow process-local isolation; failed cases retain sanitized status.

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from contextlib import ExitStack, asynccontextmanager
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

EXPECTATIONS = {"fix_required", "preserve_or_diagnose", "no_local_entity_recovery"}
DEFAULT_FIXTURE = Path(__file__).with_name("robust_faq_golden.jsonl")
MAX_CASES = 40
MAX_CANDIDATES = 125


class ReplaySafetyError(BaseException):
    """Do not let production's optional-enrichment handlers swallow isolation errors."""


class DatabaseReadFailureError(RuntimeError):
    """A production fallback cannot turn a failed read into valid benchmark evidence."""


def read_failure_record(db) -> dict:
    counts = dict(getattr(db, "read_failures", {}))
    return {"total": sum(counts.values()), "by_error_type": counts}


def reject_database_failures(artifact: dict, db) -> None:
    artifact["db_read_failures"] = read_failure_record(db)
    if artifact["db_read_failures"]["total"]:
        artifact["status"] = "invalid"
        artifact.setdefault("summary", summarize(artifact["results"]))["valid"] = False
        artifact.setdefault("error_type", "DatabaseReadFailureError")


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_cases(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows or len(rows) > MAX_CASES:
        raise ValueError("Fixture must contain 1-40 public cases")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case ID")
    for row in rows:
        if row["expectation"] not in EXPECTATIONS:
            raise ValueError("Unknown expectation")
        if not all(
            isinstance(row.get(key), str) and row[key]
            for key in ("id", "query", "family", "split")
        ):
            raise ValueError("Missing case identity/query")
        if row["expectation"] != "no_local_entity_recovery" and not row.get(
            "expected_title"
        ):
            raise ValueError("Missing positive anchor")
    return rows


def preflight(
    cases: list[dict], manifest: dict, bodies: list[dict] | None = None
) -> dict:
    """Partial metadata checks stay explicitly partial until all bodies are checked."""
    titles = manifest["titles"]
    by_name: dict[str, list[dict]] = {}
    for row in titles:
        by_name.setdefault(row["name"], []).append(row)
    content = {row["name"]: row["content"] for row in bodies or []}
    errors: list[dict] = []
    pending: list[str] = []
    anchors: dict[str, str] = {}
    for case in cases:
        title = case.get("expected_title")
        if not title:
            continue
        matches = by_name.get(title, [])
        if len(matches) != 1:
            errors.append({"id": case["id"], "reason": "missing_or_ambiguous_title"})
            continue
        anchors[case["id"]] = str(matches[0]["id"])
        if case.get("context_tokens") and title not in content:
            pending.append(case["id"])
        elif title in content and any(
            token not in content[title] for token in case.get("context_tokens", [])
        ):
            errors.append({"id": case["id"], "reason": "content_token_mismatch"})
    return {
        "valid": not errors and not pending,
        "title_valid": len(anchors)
        == sum(bool(case.get("expected_title")) for case in cases),
        "errors": errors,
        "content_checks_pending": pending,
        "anchors": anchors,
        "faq_count": len(titles),
    }


def candidate_record(candidate) -> dict:
    source = candidate.retrieval_source
    return {
        "key": candidate.key,
        "id": source.id,
        "kind": source.kind,
        "title": source.title,
        "distance": candidate.distance,
        "retrieval_path": candidate.retrieval_path,
    }


def summarize(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["execution_status"] == "completed"]
    groups = {}
    for expectation in sorted(EXPECTATIONS | {row["expectation"] for row in rows}):
        selected = [row for row in completed if row["expectation"] == expectation]
        groups[expectation] = {
            "n": len(selected),
            "families": len({row["family"] for row in selected}),
            "candidate_hits": sum(
                row["target_candidate_hit"] is True for row in selected
            ),
            "context_hits": sum(row["target_context_hit"] is True for row in selected),
            "content_passes": sum(row["content_pass"] is True for row in selected),
            "visible_hits": sum(row["target_visible_hit"] is True for row in selected),
            "zero_candidate_cases": sum(row["n_candidates"] == 0 for row in selected),
            "local_recovery_activations": sum(
                row["local_recovery_activated"] for row in selected
            ),
            "forbidden_context_hits": sum(
                row["forbidden_context_hit"] for row in selected
            ),
        }
    return {
        "attempted": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "families": len({row["family"] for row in rows}),
        "valid": len(rows) == len(completed),
        "groups": groups,
        "missing_fact_controls": sum(
            row.get("answer_fact_available") is False for row in completed
        ),
        "selected_faq_without_requested_fact": sum(
            row.get("answer_fact_available") is False
            and row["target_context_hit"] is True
            for row in completed
        ),
        "scope": "retrieval only; not generated-answer accuracy or citation entailment",
    }


async def observe_case(
    context,
    timing,
    models,
    case: dict,
    *,
    anchor: str | None,
    db,
    mode,
    case_timeout: float = 90,
    scope_catalog: dict | None = None,
) -> dict:
    """Passively capture actual candidate construction and GPU reranking inputs."""
    candidates: list = []
    pre_candidates: list = []
    filter_calls = 0
    scope_filter = getattr(context, "apply_faq_constraints_to_candidates", None)
    embedding_queries: list[str] = []
    rerank_queries: list[str] = []
    rerank_scores: list[dict] = []
    counters = {"embedding_calls": 0, "rerank_calls": 0, "build_calls": 0}
    stages: list[str] = []
    build = context._build_candidates
    embed = context._embed_variant
    rerank = context._post_rerank
    exact_reads_before = len(getattr(db, "exact_title_reads", []))
    read_failures_before = read_failure_record(db)

    def capture_build(*args, **kwargs):
        result = build(*args, **kwargs)
        counters["build_calls"] += 1
        if len(result) > MAX_CANDIDATES:
            raise ReplaySafetyError("Candidate budget exceeded")
        candidates[:] = result
        pre_candidates[:] = result
        return result

    def capture_filter(pool, constraints):
        assert scope_filter is not None
        nonlocal filter_calls
        filter_calls += 1
        if filter_calls != 1 or list(pool) != pre_candidates:
            raise ReplaySafetyError("Production scope filter capture interface changed")
        result = scope_filter(pool, constraints)
        candidates[:] = result
        return result

    async def capture_embed(text, *args, **kwargs):
        embedding_queries.append(text)
        counters["embedding_calls"] += 1
        if counters["embedding_calls"] > 3:
            raise ReplaySafetyError("Embedding call budget exceeded")
        return await embed(text, *args, **kwargs)

    async def capture_rerank(payload):
        if payload["documents"] != [candidate.rerank_text for candidate in candidates]:
            raise ReplaySafetyError("Rerank input is not the actual post-filter pool")
        counters["rerank_calls"] += 1
        if counters["rerank_calls"] > 1 or len(payload["documents"]) > MAX_CANDIDATES:
            raise ReplaySafetyError("Rerank call/candidate budget exceeded")
        rerank_queries.append(payload["query"])
        response = await rerank(payload)
        rerank_scores.extend(response.json().get("reranked_documents", []))
        return response

    timings, token = timing.start_request_timings()
    started = time.perf_counter()
    row = {key: case[key] for key in ("id", "family", "split", "expectation")}
    row.update(
        expected_title=case.get("expected_title"),
        expected_id=anchor,
        answer_fact_available=case.get("answer_fact_available"),
        fact_availability_scope=case.get("fact_availability_scope"),
        requested_transform_mode=str(mode),
    )
    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(context, "_build_candidates", capture_build)
            )
            stack.enter_context(patch.object(context, "_embed_variant", capture_embed))
            stack.enter_context(patch.object(context, "_post_rerank", capture_rerank))
            if scope_filter is not None:
                stack.enter_context(
                    patch.object(
                        context, "apply_faq_constraints_to_candidates", capture_filter
                    )
                )
            async with asyncio.timeout(case_timeout):
                result = await context.get_retrieved_context_with_sources(
                    db,
                    case["query"],
                    models.BGE_M3_LOCAL,
                    models.GPT_5_4_MINI,
                    query_transform_mode=mode,
                    credentials=None,
                    initial_k=30,
                    top_k=10,
                    on_stage=stages.append,
                )
        if counters["build_calls"] != 1:
            raise ReplaySafetyError("Production candidate capture interface changed")
        if (
            timings.lexical_search_outcome == "error"
            or read_failure_record(db)["total"] > read_failures_before["total"]
        ):
            raise DatabaseReadFailureError
        if timings.reranker_fallback_reason in {"error", "empty_or_invalid_response"}:
            raise ValueError("Reranker execution failure")
        if timings.reranker_invalid_result_count:
            raise ValueError("Invalid reranker output")
        if timings.fallback_record().get("local_faq_recovery_outcome") == "error":
            raise ValueError("Local FAQ recovery execution failure")
        target = next(
            (
                c
                for c in candidates
                if c.retrieval_source.kind == "faq" and c.retrieval_source.id == anchor
            ),
            None,
        )
        selected = set(timings.retrieval_ids)
        target_context = bool(
            target and target.key in selected and target.context_text in result.text
        )
        exact_reads = getattr(db, "exact_title_reads", [])[exact_reads_before:]
        observed_activity = bool(
            exact_reads
            or any(q != case["query"] for q in embedding_queries + rerank_queries)
        )
        recovery_record = timings.fallback_record()
        if recovery_record.get("faq_evidence_constraint_reason") == "catalog_error":
            raise ValueError("Scope catalog execution failure")
        activated = (
            recovery_record["local_faq_recovery_outcome"] == "matched"
            if "local_faq_recovery_outcome" in recovery_record
            else observed_activity
        )
        forbidden = case.get("forbidden_recovery_title")
        row.update(
            execution_status="completed",
            error_type=None,
            effective_transform_mode=result.effective_transform_mode.value,
            n_candidates=len(candidates),
            candidates=[candidate_record(c) for c in candidates],
            selected_ids=list(timings.retrieval_ids),
            visible_sources=[
                {"id": s.id, "kind": s.kind, "title": s.title} for s in result.sources
            ],
            target_candidate_hit=bool(target) if anchor else None,
            target_context_hit=target_context if anchor else None,
            target_visible_hit=any(
                s.id == anchor and s.kind == "faq" for s in result.sources
            )
            if anchor
            else None,
            content_pass=(
                target_context
                and target is not None
                and all(
                    t in target.context_text for t in case.get("context_tokens", [])
                )
            )
            if anchor and case.get("answer_fact_available") is not False
            else None,
            forbidden_context_hit=any(
                c.key in selected and c.retrieval_source.title == forbidden
                for c in candidates
            )
            if forbidden
            else False,
            local_recovery_activated=activated,
            recovery_observation="production local recovery diagnostic when available; otherwise observed exact-title reads/non-original query activity in credential-free raw path",
            local_query_activity_observed=observed_activity,
            exact_title_reads=exact_reads,
            embedding_queries=embedding_queries,
            rerank_queries=rerank_queries,
            rerank_scores=rerank_scores,
            counters=counters,
            stages=stages,
            context_characters=len(result.text),
            context_sha256=digest(result.text),
            timings=timings.as_record(),
            fallback=timings.fallback_record(),
            candidate_metrics={
                key: getattr(timings, key)
                for key in (
                    "dense_faq_candidate_count",
                    "dense_document_candidate_count",
                    "lexical_faq_candidate_count",
                    "lexical_search_outcome",
                    "final_faq_count",
                    "final_document_count",
                    "retrieval_path",
                )
            },
        )
        local = {
            key: value
            for key, value in vars(timings).items()
            if "recovery" in key or "canonical" in key
        }
        row["local_diagnostics"] = {
            key: asdict(value)
            if is_dataclass(value) and not isinstance(value, type)
            else value
            for key, value in local.items()
        }
        from tests.eval.faq_evidence_scope_checks import capture_scope_result

        active = recovery_record.get("faq_evidence_constraint_applied")
        if scope_filter is not None and not isinstance(active, bool):
            raise ReplaySafetyError("Missing actual constraint diagnostics")
        if bool(active) != bool(filter_calls):
            raise ReplaySafetyError("Constraint diagnostics/filter call mismatch")
        if scope_filter is not None and recovery_record.get(
            "faq_evidence_excluded_candidate_count"
        ) != len(pre_candidates) - len(candidates):
            raise ReplaySafetyError("Constraint exclusion count mismatch")
        row["scope_observation"] = capture_scope_result(
            before=pre_candidates,
            after=candidates,
            selected_ids=list(timings.retrieval_ids),
            sources=list(result.sources),
            context_text=result.text,
            hook_available=scope_filter is not None,
            diagnostics={
                "active": active,
                "strict_abstention": active,
                "reason": recovery_record.get("faq_evidence_constraint_reason"),
                "excluded_candidate_count": recovery_record.get(
                    "faq_evidence_excluded_candidate_count"
                ),
                "constrained_abstained": recovery_record.get(
                    "faq_evidence_constrained_abstained"
                ),
            }
            if scope_filter is not None
            else {},
            case=case,
            catalog=scope_catalog or {"facts": {}},
        )
    except (Exception, ReplaySafetyError) as exc:  # noqa: BLE001 -- retain isolated case failures without exception messages
        row.update(
            execution_status="error",
            error_type=type(exc).__name__[:80],
            counters=counters,
            fallback=timings.fallback_record(),
            failure_stage=stages[-1] if stages else "not_started",
        )
    finally:
        after = read_failure_record(db)
        row["db_read_failures"] = {
            "total": after["total"] - read_failures_before["total"],
            "by_error_type": {
                label: count - read_failures_before["by_error_type"].get(label, 0)
                for label, count in after["by_error_type"].items()
                if count > read_failures_before["by_error_type"].get(label, 0)
            },
        }
        row["latency_s"] = round(time.perf_counter() - started, 6)
        timing.reset_request_timings(token)
    return row


class ReadOnlyDatabase:
    """One read-only connection, serialized short transactions; no app lifecycle."""

    def __init__(self, connection) -> None:
        self.connection = connection
        self.lock = asyncio.Lock()
        self.exact_title_reads: list[str] = []
        self.read_failures: Counter[str] = Counter()

    @asynccontextmanager
    async def _read_transaction(self):
        try:
            async with (
                self.lock,
                self.connection.transaction(isolation="repeatable_read", readonly=True),
            ):
                yield
        except asyncio.CancelledError:
            self.read_failures["CancelledError"] += 1
            raise
        except Exception as exc:  # noqa: BLE001 -- record only class counts and preserve the original read exception
            label = re.sub(r"[^A-Za-z0-9_]", "_", type(exc).__name__[:80])
            self.read_failures[label] += 1
            raise

    async def fetch(self, query: str, *args: object):
        if (
            not re.match(r"\s*(SELECT|WITH)\b", query, re.IGNORECASE)
            and query.strip().casefold() != "show transaction_read_only"
        ):
            raise ReplaySafetyError("Only read queries are allowed")
        if (
            re.search(r"\bname\s*=\s*\$1\b", query, re.IGNORECASE)
            and args
            and isinstance(args[0], str)
        ):
            self.exact_title_reads.append(args[0])
        async with self._read_transaction():
            return await self.connection.fetch(query, *args, timeout=10)

    async def fetchrow(self, query: str, *args: object):
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *args: object):
        row = await self.fetchrow(query, *args)
        return row[0] if row else None

    async def execute(self, *_args, **_kwargs):
        raise ReplaySafetyError("Database writes are forbidden")

    async def corpus(self, expected_titles: list[str]) -> tuple[dict, list[dict]]:
        async with self._read_transaction():
            titles = await self.connection.fetch(
                "SELECT id::text, name, md5(content) AS content_hash_md5 FROM question ORDER BY id",
                timeout=10,
            )
            bodies = await self.connection.fetch(
                "SELECT name, content FROM question WHERE name = ANY($1::text[])",
                expected_titles,
                timeout=10,
            )
            # Hash full rows, including vectors/revisions, but never export those rows.
            hashes = {}
            queries = {
                "question": "SELECT id::text, md5((to_jsonb(t) - 'user_id')::text) AS row_hash FROM question AS t ORDER BY id",
                "document": "SELECT id::text, md5((to_jsonb(t) - 'user_id')::text) AS row_hash FROM document AS t ORDER BY id",
                "chunk": "SELECT id::text, md5((to_jsonb(t) - 'user_id')::text) AS row_hash FROM chunk AS t ORDER BY id",
            }
            for table, query in queries.items():
                rows = await self.connection.fetch(
                    query,
                    timeout=10,
                )
                hashes[table] = {
                    "count": len(rows),
                    "sha256": digest([dict(row) for row in rows]),
                }
        metadata = {
            "faq_count": len(titles),
            "titles": [dict(row) for row in titles],
            "row_fingerprints": hashes,
        }
        return metadata, [dict(row) for row in bodies]


def check_manifest(expected: dict, actual: dict) -> None:
    fields = ("id", "name", "content_hash_md5")

    def normalized(rows):
        return sorted(tuple(str(row[key]) for key in fields) for row in rows)

    if expected["faq_count"] != actual["faq_count"] or normalized(
        expected["titles"]
    ) != normalized(actual["titles"]):
        raise ReplaySafetyError("FAQ corpus differs from authorized manifest")


def allowed_gpu_request(method: str, url: str, gpu_url: str) -> bool:
    base = gpu_url.rstrip("/")
    return method == "POST" and url in {base + "/embeddings/embed", base + "/rerank/"}


def source_manifest(context) -> dict:
    root = Path(context.__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


def evaluation_source_manifest() -> dict:
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "robust_faq_replay.py",
            "robust_faq_compare.py",
            "faq_evidence_scope_cases.py",
            "faq_evidence_scope_checks.py",
            "faq_evidence_scope_compare.py",
        )
    }


async def run(ns: argparse.Namespace) -> dict:
    # Settings reads process environment, not a credential file. No startup hooks.
    for key, value in {
        "POSTHOG_KEY": "",
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "MCP_SERVERS": "[]",
        "MCP_HTTP_URLS": "",
        "MCP_SSE_URLS": "",
    }.items():
        os.environ[key] = value
    logging.disable(logging.CRITICAL)
    import asyncpg
    import httpx

    from app.llms import context
    from app.llms.models import (
        BGE_M3_EMBEDDING_SPEC_VERSION,
        MODEL_DISTANCE_THRESHOLDS,
        Model,
    )
    from app.llms.query_modes import QueryTransformMode
    from app.utils import http_client, timing

    scope_catalog = None
    if getattr(ns, "scope_catalog", None):
        from tests.eval.faq_evidence_scope_cases import (
            load_fact_catalog,
            load_scope_cases,
            preflight_scope_cases,
        )

        scope_catalog = load_fact_catalog(Path(ns.scope_catalog))
        cases = load_scope_cases(Path(ns.fixture), scope_catalog)
    else:
        cases = load_cases(Path(ns.fixture))
    expected = json.loads(Path(ns.corpus_manifest).read_text(encoding="utf-8-sig"))
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise ReplaySafetyError("DATABASE_URL must be inherited in memory")
    settings = context.settings
    modes = {
        "raw": QueryTransformMode.RAW,
        "credential_missing": QueryTransformMode.REWRITE_HYDE,
    }
    if (settings.RERANKER_MIN_SCORE, settings.SOURCE_RERANKER_MIN_SCORE) != (0.1, 0.3):
        raise ReplaySafetyError("Unexpected context/source floors")
    calls: Counter = Counter()
    http_failures: Counter = Counter()
    send = httpx.AsyncClient.send

    async def guarded_send(client, request, *args, **kwargs):
        if not allowed_gpu_request(
            request.method, str(request.url), settings.GPU_API_URL
        ):
            raise ReplaySafetyError("Non-GPU HTTP request forbidden")
        calls[request.url.path] += 1
        if sum(calls.values()) > len(cases) * 4:
            raise ReplaySafetyError("GPU request budget exceeded")
        request.extensions["timeout"] = dict.fromkeys(
            ("connect", "read", "write", "pool"), 30.0
        )
        try:
            response = await send(client, request, *args, **kwargs)
        except httpx.HTTPError, TimeoutError:
            http_failures[request.url.path] += 1
            raise
        if response.status_code >= 400:
            http_failures[request.url.path] += 1
        return response

    def forbidden_sync(*_args, **_kwargs):
        raise ReplaySafetyError("Synchronous HTTP forbidden")

    connection = await asyncpg.connect(
        dsn,
        timeout=15,
        server_settings={
            "default_transaction_read_only": "on",
            "statement_timeout": "10000",
            "idle_in_transaction_session_timeout": "15000",
        },
    )
    db = ReadOnlyDatabase(connection)
    artifact: dict = {
        "schema_version": "robust-faq-replay-v1",
        "revision_label": ns.revision_label,
        "status": "invalid",
        "results": [],
        "selection": {
            "cases": len(cases),
            "families": len({case["family"] for case in cases}),
            "case_ids": [case["id"] for case in cases],
        },
    }
    http_client.init_http_client()
    try:
        if await db.fetchval("SHOW transaction_read_only") != "on":
            raise ReplaySafetyError("Read-only enforcement unavailable")
        wanted = sorted(
            {case["expected_title"] for case in cases if case.get("expected_title")}
            | (set(scope_catalog["sources"]) if scope_catalog else set())
        )
        before, bodies = await db.corpus(wanted)
        check_manifest(expected, before)
        if scope_catalog:
            checked = preflight_scope_cases(cases, scope_catalog, before, bodies)
            checked["anchors"] = {
                case["id"]: checked["resolved_titles"].get(case.get("expected_title"))
                for case in cases
                if case.get("expected_title")
            }
        else:
            checked = preflight(cases, before, bodies)
        artifact["preflight"] = checked
        if not checked["valid"]:
            raise ReplaySafetyError("Fixture corpus preflight failed")
        sources = source_manifest(context)
        eval_sources = evaluation_source_manifest()
        artifact["manifest"] = {
            "fixture_sha256": hashlib.sha256(Path(ns.fixture).read_bytes()).hexdigest(),
            "gpu_manifest_sha256": hashlib.sha256(
                Path(ns.gpu_manifest).read_bytes()
            ).hexdigest(),
            "corpus": before,
            "source_sha256": sources,
            "evaluation_source_sha256": eval_sources,
            "embedding_model": Model.BGE_M3_LOCAL.value,
            "bge_spec": BGE_M3_EMBEDDING_SPEC_VERSION,
            "embedding_dimensions": 1024,
            "python": sys.version,
            "dependencies": {
                name: version(name) for name in ("asyncpg", "httpx", "pydantic")
            },
            "dense_cutoff": MODEL_DISTANCE_THRESHOLDS[Model.BGE_M3_LOCAL],
            "context_floor": 0.1,
            "source_floor": 0.3,
            "initial_k": 30,
            "top_k": 10,
            "execution_mode": ns.mode,
            "credentials_supplied": False,
            "transport": "actual production retrieval; GPU-only HTTP; sequential; no generation",
            "fingerprint_note": "FAQ rows match supplied manifest; full-row hashed fingerprints are runner-specific and include vectors/revisions. Short read-only transactions do not share one snapshot across GPU calls; before/after drift invalidates run.",
        }
        if scope_catalog:
            artifact["manifest"]["scope_catalog_sha256"] = hashlib.sha256(
                Path(ns.scope_catalog).read_bytes()
            ).hexdigest()
        with (
            patch.object(httpx.AsyncClient, "send", guarded_send),
            patch.object(httpx.Client, "send", forbidden_sync),
        ):
            for case in cases:
                row = await observe_case(
                    context,
                    timing,
                    Model,
                    case,
                    anchor=checked["anchors"].get(case["id"]),
                    db=db,
                    mode=modes[ns.mode],
                    scope_catalog=scope_catalog,
                )
                artifact["results"].append(row)
                Path(ns.output).write_text(
                    json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                if row.get("error_type") == "ReplaySafetyError":
                    break
        after, _bodies = await db.corpus(wanted)
        if (
            before != after
            or sources != source_manifest(context)
            or eval_sources != evaluation_source_manifest()
        ):
            raise ReplaySafetyError("Corpus/source changed during replay")
        artifact["summary"] = summarize(artifact["results"])
        artifact["summary"]["selected_cases"] = len(cases)
        artifact["summary"]["not_attempted"] = len(cases) - len(artifact["results"])
        if len(artifact["results"]) != len(cases):
            artifact["summary"]["valid"] = False
        if http_failures:
            artifact["summary"]["valid"] = False
            artifact["error_type"] = "GpuTransportFailure"
        artifact["status"] = "completed" if artifact["summary"]["valid"] else "invalid"
    except (Exception, ReplaySafetyError) as exc:  # noqa: BLE001 -- preserve incomplete public replay artifacts
        artifact["error_type"] = type(exc).__name__[:80]
    finally:
        reject_database_failures(artifact, db)
        artifact["gpu_http_calls"] = dict(calls)
        artifact["gpu_http_failures"] = dict(http_failures)
        Path(ns.output).write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        await http_client.close_http_client()
        await connection.close(timeout=10)
    return artifact


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    cli.add_argument(
        "--scope-catalog",
        help="Public scope catalog; enables separate scope fixture schema and strict preflight",
    )
    cli.add_argument("--corpus-manifest", required=True)
    cli.add_argument("--revision-label", required=True)
    cli.add_argument(
        "--gpu-manifest",
        required=True,
        help="Parent-verified public GPU/model identity artifact; only its hash is saved",
    )
    cli.add_argument("--mode", choices=("raw", "credential_missing"), default="raw")
    cli.add_argument("--output", required=True)
    return cli


def main() -> int:
    ns = parser().parse_args()
    if Path(ns.output).exists():
        raise SystemExit("Refusing to overwrite an existing replay artifact")

    async def bounded():
        async with asyncio.timeout(600):
            return await run(ns)

    try:
        result = asyncio.run(bounded())
    except (Exception, ReplaySafetyError) as exc:  # noqa: BLE001 -- CLI reports only sanitized failure classes
        error = {
            "status": "invalid",
            "error_type": type(exc).__name__[:80],
            "results": [],
        }
        if Path(ns.output).exists():
            error.update(json.loads(Path(ns.output).read_text(encoding="utf-8")))
            error.update(status="invalid", error_type=type(exc).__name__[:80])
        Path(ns.output).write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(json.dumps({"status": "invalid", "error_type": type(exc).__name__[:80]}))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result.get("summary"),
                "error_type": result.get("error_type"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
