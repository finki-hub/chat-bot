from contextlib import asynccontextmanager
from uuid import UUID

import anyio

from app.data.connection import Database
from app.data.embedding_lifecycle import EmbeddingCandidate
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.embedding_worker_drain import drain_dirty_embeddings


def test_drain_leaves_dirty_rows_unwritten_when_generation_fails(monkeypatch) -> None:
    async def run() -> None:
        # Given: one captured dirty row and a provider that exhausts its retries.
        candidate = EmbeddingCandidate(
            corpus=EmbeddingCorpus.QUESTION,
            id=UUID("00000000-0000-0000-0000-000000000401"),
            text="not logged",
            revision=1,
        )
        writes = 0

        async def fetch_dirty(_database, corpus, _limit):
            return (candidate,) if corpus is EmbeddingCorpus.QUESTION else ()

        async def generate(_texts, _model, **_kwargs):
            raise ConnectionError("provider unavailable")

        async def persist(_database, _batch):
            nonlocal writes
            writes += 1
            raise AssertionError("failed provider output must not be persisted")

        monkeypatch.setattr(
            "app.embedding_worker_drain.fetch_dirty_embeddings",
            fetch_dirty,
        )
        monkeypatch.setattr(
            "app.embedding_worker_drain.persist_embedding_batch",
            persist,
        )

        # When: the worker drain reaches the failed provider batch.
        report = await drain_dirty_embeddings(
            Database("postgresql://worker-test"),
            generate,
        )

        # Then: it records the failure and retains the durable dirty row for a future wake.
        assert report.failed_batches == 1
        assert report.updated_rows == 0
        assert writes == 0

    anyio.run(run)


def test_drain_continues_to_later_corpora_after_generation_fails(monkeypatch) -> None:
    async def run() -> None:
        # Given: the first corpus has a poison batch and a later corpus has work.
        poison = EmbeddingCandidate(
            corpus=EmbeddingCorpus.QUESTION,
            id=UUID("00000000-0000-0000-0000-000000000411"),
            text="poison",
            revision=1,
        )
        later = EmbeddingCandidate(
            corpus=EmbeddingCorpus.CHUNK,
            id=UUID("00000000-0000-0000-0000-000000000412"),
            text="later",
            revision=1,
        )
        persisted: list[EmbeddingCorpus] = []
        attempts: dict[EmbeddingCorpus, int] = {
            EmbeddingCorpus.QUESTION: 0,
            EmbeddingCorpus.CHUNK: 0,
        }

        async def fetch_dirty(_database, corpus, _limit):
            match corpus:
                case EmbeddingCorpus.QUESTION:
                    attempts[corpus] += 1
                    return (poison,)
                case EmbeddingCorpus.CHUNK:
                    attempts[corpus] += 1
                    return (later,) if attempts[corpus] == 1 else ()
                case EmbeddingCorpus.DIPLOMA | EmbeddingCorpus.PROFESSOR_DOCUMENT:
                    return ()

        async def generate(texts, _model, **_kwargs):
            if texts == ["poison"]:
                raise ConnectionError("provider unavailable")
            return [[0.0] * 1024]

        async def persist(_database, batch):
            persisted.append(batch.corpus)

            class Result:
                valid = True
                updated = len(batch.candidates)

            return Result()

        monkeypatch.setattr(
            "app.embedding_worker_drain.fetch_dirty_embeddings",
            fetch_dirty,
        )
        monkeypatch.setattr(
            "app.embedding_worker_drain.persist_embedding_batch",
            persist,
        )

        # When: the drain hits the poison batch before later corpus work.
        report = await drain_dirty_embeddings(
            Database("postgresql://worker-test"),
            generate,
        )

        # Then: the failed batch is retained, but later durable work still progresses.
        assert report.failed_batches == 1
        assert report.updated_rows == 1
        assert persisted == [EmbeddingCorpus.CHUNK]
        assert attempts[EmbeddingCorpus.QUESTION] == 1

    anyio.run(run)


def test_drain_rejects_malformed_vectors_without_writing(monkeypatch) -> None:
    async def run() -> None:
        # Given: one dirty row and a provider response with the wrong vector dimension.
        candidate = EmbeddingCandidate(
            corpus=EmbeddingCorpus.QUESTION,
            id=UUID("00000000-0000-0000-0000-000000000402"),
            text="not logged",
            revision=1,
        )

        async def fetch_dirty(_database, corpus, _limit):
            return (candidate,) if corpus is EmbeddingCorpus.QUESTION else ()

        async def generate(_texts, _model, **_kwargs):
            return [[0.0] * 3]

        monkeypatch.setattr(
            "app.embedding_worker_drain.fetch_dirty_embeddings",
            fetch_dirty,
        )
        database = Database("postgresql://worker-test")

        @asynccontextmanager
        async def unexpected_transaction():
            raise AssertionError(
                "malformed provider output must not open a transaction",
            )
            yield

        monkeypatch.setattr(database, "transaction", unexpected_transaction)

        # When: the malformed provider response reaches the worker drain.
        report = await drain_dirty_embeddings(database, generate)

        # Then: batch validation prevents persistence and leaves the candidate dirty.
        assert report.invalid_batches == 1
        assert report.updated_rows == 0

    anyio.run(run)
