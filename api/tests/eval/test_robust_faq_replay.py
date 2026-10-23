import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest
from asyncpg import PostgresError

from tests.eval import robust_faq_replay as replay


def case():
    return {
        "id": "test-public",
        "family": "service",
        "split": "control",
        "expectation": "fix_required",
        "query": "public support test",
        "expected_title": "Студентска служба",
        "context_tokens": ["ТМФ", "117"],
    }


def test_public_fixture_counts_and_families():
    rows = replay.load_cases(replay.DEFAULT_FIXTURE)
    assert len(rows) == 38
    assert len({row["family"] for row in rows}) == 25
    assert sum(row["expectation"] == "fix_required" for row in rows) == 8
    assert sum(row["expectation"] == "no_local_entity_recovery" for row in rows) == 18
    assert sum(row.get("answer_fact_available") is False for row in rows) == 2


@pytest.mark.parametrize(
    "public_case",
    [
        row
        for row in replay.load_cases(replay.DEFAULT_FIXTURE)
        if row["expectation"] == "fix_required"
    ],
    ids=lambda row: row["id"],
)
def test_required_public_queries_use_shared_resolver(public_case):
    from app.llms.faq_query_recovery import resolve_faq_query  # noqa: PLC0415 -- pure production helper, no network

    titles = sorted(
        {
            row["expected_title"]
            for row in replay.load_cases(replay.DEFAULT_FIXTURE)
            if row.get("expected_title")
        }
    )
    resolved = resolve_faq_query(public_case["query"], titles)
    assert resolved is not None
    assert resolved.title == public_case["expected_title"]


def test_synthetic_catalog_controls_use_shared_resolver():
    from app.llms.faq_query_recovery import resolve_faq_query  # noqa: PLC0415 -- pure production helper, no network

    path = replay.DEFAULT_FIXTURE.with_name("robust_faq_catalog_controls.jsonl")
    controls = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(controls) == 4
    for control in controls:
        assert resolve_faq_query(control["query"], control["titles"]) is None, control[
            "id"
        ]


def test_preflight_partial_content_stays_pending():
    title = case()["expected_title"]
    manifest = {"titles": [{"name": title, "id": "public-id"}]}
    checked = replay.preflight([case()], manifest)
    assert checked["title_valid"] is True
    assert checked["valid"] is False
    assert checked["content_checks_pending"] == ["test-public"]
    checked = replay.preflight(
        [case()], manifest, [{"name": title, "content": "ТМФ 117"}]
    )
    assert checked["valid"] is True


@pytest.mark.parametrize(
    "titles",
    [
        [],
        [
            {"name": "Студентска служба", "id": "a"},
            {"name": "Студентска служба", "id": "b"},
        ],
    ],
)
def test_preflight_rejects_missing_or_duplicate_title(titles):
    checked = replay.preflight([case()], {"titles": titles})
    assert checked["valid"] is False
    assert checked["title_valid"] is False


def test_preflight_rejects_missing_public_fact():
    checked = replay.preflight(
        [case()],
        {"titles": [{"name": "Студентска служба", "id": "a"}]},
        [{"name": "Студентска служба", "content": "ТМФ"}],
    )
    assert checked["errors"] == [
        {"id": "test-public", "reason": "content_token_mismatch"}
    ]


@pytest.mark.parametrize(
    ("method", "url", "allowed"),
    [
        ("POST", "http://gpu-api:8888/embeddings/embed", True),
        ("POST", "http://gpu-api:8888/rerank/", True),
        ("GET", "http://gpu-api:8888/rerank/", False),
        ("POST", "http://gpu-api:8888/rerank/?key=secret", False),
        ("POST", "https://provider.invalid/chat/completions", False),
        ("POST", "http://gpu-api:8888.evil.invalid/rerank/", False),
    ],
)
def test_gpu_allowlist(method, url, allowed):
    assert replay.allowed_gpu_request(method, url, "http://gpu-api:8888") is allowed


def test_manifest_drift_rejected():
    expected = {
        "faq_count": 1,
        "titles": [{"id": "a", "name": "FAQ", "content_hash_md5": "old"}],
    }
    actual = {
        "faq_count": 1,
        "titles": [{"id": "a", "name": "FAQ", "content_hash_md5": "new"}],
    }
    with pytest.raises(replay.ReplaySafetyError):
        replay.check_manifest(expected, actual)


def test_database_write_methods_blocked():
    db = replay.ReadOnlyDatabase(SimpleNamespace())

    async def run():
        with pytest.raises(replay.ReplaySafetyError):
            await db.execute("DELETE FROM question")
        with pytest.raises(replay.ReplaySafetyError):
            await db.fetch("UPDATE question SET name='bad'")

    anyio.run(run)


def test_read_queries_use_serialized_readonly_transactions():
    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    calls = []

    def transaction(**kwargs):
        calls.append(kwargs)
        return Transaction()

    connection = SimpleNamespace(
        transaction=transaction, fetch=AsyncMock(return_value=[("on",)])
    )
    db = replay.ReadOnlyDatabase(connection)

    async def run():
        assert await db.fetchval("SHOW transaction_read_only") == "on"
        await db.fetch("SELECT name FROM question WHERE name = $1", "Студентска служба")

    anyio.run(run)
    assert calls == [{"isolation": "repeatable_read", "readonly": True}] * 2
    assert db.exact_title_reads == ["Студентска служба"]
    assert connection.fetch.await_args.kwargs["timeout"] == 10


@pytest.mark.parametrize("score", [0.9, 0.0123])
def test_observer_runs_actual_mocked_production_selection(monkeypatch, score):
    from app.llms import context  # noqa: PLC0415 -- actual production boundary test
    from app.llms.models import Model  # noqa: PLC0415
    from app.schemas.questions import QuestionSchema  # noqa: PLC0415
    from app.utils import timing  # noqa: PLC0415

    faq = QuestionSchema(
        id=uuid4(),
        name="Студентска служба",
        content="ТМФ, до кабинетот 117.",
        links={},
        distance=0.2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    monkeypatch.setattr(context, "_embed_variant", AsyncMock(return_value=[1.0]))
    monkeypatch.setattr(context, "_search_both", AsyncMock(return_value=([faq], [])))
    monkeypatch.setattr(context, "get_matching_questions", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        context,
        "_post_rerank",
        AsyncMock(
            return_value=SimpleNamespace(
                json=lambda: {"reranked_documents": [{"index": 0, "score": score}]}
            )
        ),
    )

    async def render(_db, final, _model):
        return "\n".join(c.context_text for c in final)

    monkeypatch.setattr(context, "_expand_and_render", render)
    db = SimpleNamespace(exact_title_reads=[], fetch=AsyncMock(return_value=[]))

    async def run():
        return await replay.observe_case(
            context,
            timing,
            Model,
            case(),
            anchor=str(faq.id),
            db=db,
        )

    row = anyio.run(run)
    assert row["execution_status"] == "completed"
    assert row["n_candidates"] == 1
    assert row["target_candidate_hit"] is True
    assert row["target_context_hit"] is True
    assert row["content_pass"] is True
    assert row["target_visible_hit"] is (score >= 0.3)
    assert row["local_recovery_activated"] is False
    assert row["fallback"]["reranker_fallback_reason"] == (
        "none" if score >= 0.1 else "score_floor"
    )
    assert timing._current.get() is None


def test_loader_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "duplicate.jsonl"

    path.write_text(json.dumps(case()) + "\n" + json.dumps(case()), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        replay.load_cases(path)


def read_test_database(error, operation="fetch"):
    class Transaction:
        async def __aenter__(self):
            if operation == "begin":
                raise error
            return self

        async def __aexit__(self, *_args):
            if operation == "end":
                raise error
            return False

    connection = SimpleNamespace(
        transaction=lambda **_kwargs: Transaction(),
        fetch=AsyncMock(side_effect=error)
        if operation in {"fetch", "corpus"}
        else AsyncMock(return_value=[]),
    )
    return replay.ReadOnlyDatabase(connection)


@pytest.mark.parametrize("operation", ["fetch", "begin", "end", "corpus"])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
def test_read_and_transaction_errors_record_only_class_and_preserve_original(
    operation, error_type
):
    error = error_type("SENSITIVE_EXCEPTION_MARKER")
    db = read_test_database(error, operation)

    async def run():
        read = (
            db.corpus([])
            if operation == "corpus"
            else db.fetch("SELECT SENSITIVE_SQL_MARKER", "SENSITIVE_ARGUMENT_MARKER")
        )
        with pytest.raises(error_type, match="SENSITIVE_EXCEPTION_MARKER") as caught:
            await read
        assert caught.value is error

    anyio.run(run)
    record = replay.read_failure_record(db)
    assert record == {"total": 1, "by_error_type": {error_type.__name__: 1}}
    assert "SENSITIVE" not in json.dumps(record)


def test_cancelled_read_is_recorded_and_original_cancellation_propagates():
    error = asyncio.CancelledError()
    db = read_test_database(error)

    async def run():
        with pytest.raises(asyncio.CancelledError) as caught:
            await db.fetch("SELECT ignored")
        assert caught.value is error

    anyio.run(run)
    assert replay.read_failure_record(db) == {
        "total": 1,
        "by_error_type": {"CancelledError": 1},
    }


def test_global_read_failure_gate_cannot_accept_successful_case_metadata():
    db = read_test_database(RuntimeError())
    db.read_failures["RuntimeError"] = 1
    artifact: dict = {"status": "completed", "results": [], "summary": {"valid": True}}
    replay.reject_database_failures(artifact, db)
    assert artifact["status"] == "invalid"
    assert artifact["summary"]["valid"] is False
    assert artifact["db_read_failures"]["total"] == 1
    clean = replay.ReadOnlyDatabase(SimpleNamespace())
    healthy: dict = {"status": "completed", "results": [], "summary": {"valid": True}}
    replay.reject_database_failures(healthy, clean)
    assert healthy["status"] == "completed"
    assert healthy["summary"]["valid"] is True


@pytest.mark.parametrize(
    "stage", ["lexical_without_adapter", "canonical_timeout", "window"]
)
def test_actual_production_swallowed_read_faults_invalidate_replay(monkeypatch, stage):
    from app.llms import context  # noqa: PLC0415 -- actual production function with offline dependencies
    from app.llms.faq_query_recovery import resolve_faq_query  # noqa: PLC0415
    from app.llms.faq_query_recovery_plan import FaqQueryRecoveryPlan  # noqa: PLC0415
    from app.llms.models import Model  # noqa: PLC0415
    from app.schemas.documents import ChunkSchema  # noqa: PLC0415
    from app.schemas.questions import QuestionSchema  # noqa: PLC0415
    from app.utils import timing  # noqa: PLC0415

    public_case = case()
    faq = QuestionSchema(
        id=uuid4(),
        name=public_case["expected_title"],
        content="ТМФ 117",
        links={},
        distance=0.2,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    chunks = []
    db = read_test_database(
        TimeoutError("SENSITIVE_EXCEPTION_MARKER")
        if stage == "canonical_timeout"
        else PostgresError("SENSITIVE_EXCEPTION_MARKER")
    )
    canonical_query: str | None = None
    if stage == "canonical_timeout":
        public_case["query"] = "Kade e sluzba"
        recovery = resolve_faq_query(public_case["query"], [faq.name])
        assert recovery is not None
        canonical_query = recovery.canonical_query
        monkeypatch.setattr(
            context,
            "plan_faq_query_recovery",
            AsyncMock(
                return_value=FaqQueryRecoveryPlan(
                    outcome="matched", recovery=recovery, question=faq
                )
            ),
        )
    else:
        monkeypatch.setattr(
            context,
            "plan_faq_query_recovery",
            AsyncMock(return_value=FaqQueryRecoveryPlan(outcome="skipped")),
        )
    if stage == "window":
        chunks.append(
            ChunkSchema(
                id=uuid4(),
                document_id=uuid4(),
                document_name="public-rules",
                document_title="Public rules",
                chunk_index=0,
                content="Public excerpt.",
                distance=0.2,
            )
        )

        async def window(*_args, **_kwargs):
            return await db.fetch(
                "SELECT SENSITIVE_SQL_MARKER", "SENSITIVE_ARGUMENT_MARKER"
            )

        monkeypatch.setattr(context, "get_chunks_window", window)

    async def embed(text, *_args, **_kwargs):
        return [0.2] if text == canonical_query else [0.1]

    async def search(_db, embedding, *_args, **_kwargs):
        if stage == "canonical_timeout" and embedding == [0.2]:
            await db.fetch("SELECT SENSITIVE_SQL_MARKER", "SENSITIVE_ARGUMENT_MARKER")
        return [faq], chunks

    if stage == "lexical_without_adapter":
        monkeypatch.setattr(
            context,
            "get_matching_questions",
            AsyncMock(side_effect=PostgresError("SENSITIVE_EXCEPTION_MARKER")),
        )
    else:
        monkeypatch.setattr(
            context, "get_matching_questions", AsyncMock(return_value=[])
        )
    rerank = AsyncMock(
        return_value=SimpleNamespace(
            json=lambda: {
                "reranked_documents": [
                    {"index": index, "score": 0.9} for index in range(1 + len(chunks))
                ]
            }
        )
    )
    monkeypatch.setattr(context, "_embed_variant", embed)
    monkeypatch.setattr(context, "_search_both", search)
    monkeypatch.setattr(context, "_post_rerank", rerank)

    async def run():
        return await replay.observe_case(
            context,
            timing,
            Model,
            public_case,
            anchor=str(faq.id),
            db=db,
        )

    row = anyio.run(run)
    rerank.assert_awaited_once()
    assert row["execution_status"] == "error"
    assert row["error_type"] == "DatabaseReadFailureError"
    assert row["db_read_failures"]["total"] == (
        0 if stage == "lexical_without_adapter" else 1
    )
    assert "SENSITIVE" not in json.dumps(row)
