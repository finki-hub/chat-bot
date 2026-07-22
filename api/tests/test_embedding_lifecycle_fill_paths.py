import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID

import anyio
from anyio.lowlevel import checkpoint
from asyncpg import PostgresError
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms.embeddings import stream_fill_embeddings
from app.llms.models import Model


class FillState:
    def __init__(self, *, apply_write: bool) -> None:
        self.apply_write = apply_write
        self.calls: list[tuple[str, tuple[str | int | UUID, ...]]] = []

    def fetch(self, query, *args):
        self.calls.append((query, args))
        return [
            {
                "id": UUID("00000000-0000-0000-0000-000000000201"),
                "name": "Question",
                "content": "Answer",
                "embedding_revision": 7,
            },
        ]

    def execute(self, query, *args):
        self.calls.append((query, args))
        return "UPDATE 1"

    async def fetchval(self, query, *args):
        await checkpoint()
        self.calls.append((query, args))
        return True if self.apply_write else None

    def transaction(self) -> FillTransaction:
        return FillTransaction(self)


class FillTransaction:
    def __init__(self, state: FillState) -> None:
        self._state = state

    async def __aenter__(self) -> FillState:
        await checkpoint()
        return self._state

    async def __aexit__(self, *_args) -> None:
        await checkpoint()


class MixedFillState(FillState):
    def __init__(self) -> None:
        super().__init__(apply_write=True)
        self.apply_results = iter((True, None))

    def fetch(self, query, *args):
        self.calls.append((query, args))
        return [
            {
                "id": UUID("00000000-0000-0000-0000-000000000202"),
                "name": "Current",
                "content": "Answer",
                "embedding_revision": 7,
            },
            {
                "id": UUID("00000000-0000-0000-0000-000000000203"),
                "name": "Stale",
                "content": "Answer",
                "embedding_revision": 7,
            },
        ]

    async def fetchval(self, query, *args):
        await checkpoint()
        self.calls.append((query, args))
        return next(self.apply_results)


def _database(state, monkeypatch) -> Database:
    database = Database("postgresql://lifecycle-test")
    monkeypatch.setattr(database, "fetch", AsyncMock(side_effect=state.fetch))
    monkeypatch.setattr(database, "execute", AsyncMock(side_effect=state.execute))
    monkeypatch.setattr(database, "fetchval", state.fetchval)
    monkeypatch.setattr(database, "transaction", state.transaction)
    return database


async def _sse_events(response: StreamingResponse) -> list[str]:
    events: list[str] = []
    async for chunk in response.body_iterator:
        text = chunk if isinstance(chunk, str) else bytes(chunk).decode("utf-8")
        events.extend(
            line.removeprefix("data:").strip()
            for line in text.splitlines()
            if line.startswith("data:")
        )
    return events


def test_embedding_fills_imports_without_the_embeddings_facade() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import app.llms.embedding_fills"],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_manual_question_fill_writes_lifecycle_metadata_with_captured_revision(
    monkeypatch,
) -> None:
    generate = AsyncMock(return_value=[[0.0] * 1024])

    async def run() -> None:
        # Given: a current source revision and a deterministic BGE provider.
        state = FillState(apply_write=True)
        database = _database(state, monkeypatch)
        monkeypatch.setattr("app.llms.embedding_fills.generate_embeddings", generate)

        # When: the actual compatibility SSE fill generator is consumed.
        response = stream_fill_embeddings(
            database,
            Model.BGE_M3_LOCAL,
            questions=["Question"],
        )
        events = await _sse_events(response)

        # Then: persistence is revision-guarded and writes vector/version/timestamp together.
        writes = [query for query, _args in state.calls if "UPDATE question" in query]
        assert len(writes) == 1
        assert "embedding_bge_m3_version" in writes[0]
        assert "embedding_bge_m3_updated_at" in writes[0]
        assert "embedding_revision = $4" in writes[0]
        assert '"status": "ok"' in events[0]

    anyio.run(run)


def test_manual_question_fill_reports_no_success_when_source_revision_races(
    monkeypatch,
) -> None:
    generate = AsyncMock(return_value=[[0.0] * 1024])

    async def run() -> None:
        # Given: a source update that advances the revision before guarded persistence.
        state = FillState(apply_write=False)
        database = _database(state, monkeypatch)
        monkeypatch.setattr("app.llms.embedding_fills.generate_embeddings", generate)

        # When: the real SSE fill generator completes the stale batch.
        response = stream_fill_embeddings(
            database,
            Model.BGE_M3_LOCAL,
            questions=["Question"],
        )
        events = await _sse_events(response)

        # Then: no stale persistence is reported as a successful fill.
        assert '"status": "ok"' not in events[0]

    anyio.run(run)


def test_manual_question_fill_reports_mixed_guarded_batch_outcomes(monkeypatch) -> None:
    generate = AsyncMock(return_value=[[0.0] * 1024, [1.0] * 1024])

    async def run() -> None:
        state = MixedFillState()
        database = _database(state, monkeypatch)
        monkeypatch.setattr("app.llms.embedding_fills.generate_embeddings", generate)

        response = stream_fill_embeddings(
            database,
            Model.BGE_M3_LOCAL,
            questions=["Current", "Stale"],
        )
        events = await _sse_events(response)

        assert '"status": "ok"' in events[0]
        assert '"status": "error"' in events[1]

    anyio.run(run)


def test_manual_question_fill_reports_persistence_failure_as_row_error(
    monkeypatch,
) -> None:
    generate = AsyncMock(return_value=[[0.0] * 1024])
    fail_persist = AsyncMock(side_effect=PostgresError("database unavailable"))

    async def run() -> None:
        # Given: embedding generation succeeds but database persistence fails.
        state = FillState(apply_write=True)
        database = _database(state, monkeypatch)
        monkeypatch.setattr("app.llms.embedding_fills.generate_embeddings", generate)
        monkeypatch.setattr(
            "app.llms.embedding_fills.persist_embedding_batch",
            fail_persist,
        )

        # When: the real SSE fill generator is consumed.
        response = stream_fill_embeddings(
            database,
            Model.BGE_M3_LOCAL,
            questions=["Question"],
        )
        events = await _sse_events(response)

        # Then: the stream completes with a row-level error event.
        assert len(events) == 1
        assert '"status": "error"' in events[0]
        assert "Embedding processing failed." in events[0]

    anyio.run(run)
