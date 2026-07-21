from dataclasses import dataclass
from typing import Final, assert_never
from uuid import UUID

from asyncpg import Record
from asyncpg.pool import PoolConnectionProxy

from app.data.connection import Database
from app.data.embedding_lifecycle_sql import (
    COUNT_SQL,
    DIRTY_SELECT_SQL,
    PERSIST_SQL,
    REBUILD_SQL,
    EmbeddingCorpus,
)
from app.llms.models import (
    BGE_M3_EMBEDDING_SPEC_VERSION,
    MODEL_EMBEDDING_DIMENSIONS,
    Model,
)
from app.recommenders.text import build_proposal_text
from app.utils.database import embedding_to_pgvector

BGE_M3_DIMENSIONS: Final = MODEL_EMBEDDING_DIMENSIONS[Model.BGE_M3]
EMBEDDING_WAKE_PAYLOAD: Final = "wake"


@dataclass(frozen=True, slots=True)
class EmbeddingCandidate:
    """A dirty corpus row captured with its source revision."""

    corpus: EmbeddingCorpus
    id: UUID
    text: str
    revision: int


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """One provider result batch for a single corpus."""

    corpus: EmbeddingCorpus
    candidates: tuple[EmbeddingCandidate, ...]
    vectors: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class EmbeddingWriteResult:
    """The outcome of validating and writing a provider batch."""

    valid: bool
    updated: int


@dataclass(frozen=True, slots=True)
class EmbeddingLifecycleCount:
    """Current and dirty row totals for one corpus."""

    corpus: EmbeddingCorpus
    ready: int
    dirty: int


def canonical_question_text(name: str, content: str) -> str:
    """Build the canonical BGE-M3 input for a question."""
    return f"Наслов: {name}\nСодржина: {content}"


def canonical_chunk_text(title: str, section: str | None, content: str) -> str:
    """Build the canonical BGE-M3 input for a document chunk."""
    title_with_section = f"{title} ({section})" if section else title
    return f"Наслов: {title_with_section}\nСодржина: {content}"


def canonical_diploma_text(title: str, description: str | None) -> str:
    """Build the canonical BGE-M3 input for a diploma."""
    return build_proposal_text(title, description)


def canonical_professor_document_text(title: str, abstract: str | None) -> str:
    """Build the canonical BGE-M3 input for a professor document."""
    return build_proposal_text(title, abstract)


def validate_embedding_batch(batch: EmbeddingBatch) -> bool:
    """Return whether a provider batch is complete and has 1024-dimensional vectors."""
    return len(batch.candidates) == len(batch.vectors) and all(
        candidate.corpus == batch.corpus and len(vector) == BGE_M3_DIMENSIONS
        for candidate, vector in zip(batch.candidates, batch.vectors, strict=True)
    )


async def fetch_dirty_embeddings(
    database: Database,
    corpus: EmbeddingCorpus,
    limit: int,
) -> tuple[EmbeddingCandidate, ...]:
    """Fetch one deterministic bounded dirty batch from an allowlisted corpus."""
    rows = await database.fetch(
        DIRTY_SELECT_SQL[corpus],
        BGE_M3_EMBEDDING_SPEC_VERSION,
        limit,
    )
    return tuple(_candidate_from_row(corpus, row) for row in rows)


async def lifecycle_counts(database: Database) -> tuple[EmbeddingLifecycleCount, ...]:
    """Return ready and dirty BGE-M3 row counts for every corpus."""
    counts: list[EmbeddingLifecycleCount] = []
    for corpus in EmbeddingCorpus:
        row = await database.fetchrow(
            COUNT_SQL[corpus],
            BGE_M3_EMBEDDING_SPEC_VERSION,
        )
        if row is None:
            msg = f"Missing lifecycle count for {corpus.value}"
            raise RuntimeError(msg)
        counts.append(
            EmbeddingLifecycleCount(
                corpus=corpus,
                ready=int(row["ready"]),
                dirty=int(row["dirty"]),
            ),
        )
    return tuple(counts)


async def persist_embedding_batch(
    database: Database,
    batch: EmbeddingBatch,
) -> EmbeddingWriteResult:
    """Validate and revision-guardedly persist one BGE-M3 provider batch."""
    if not validate_embedding_batch(batch):
        return EmbeddingWriteResult(valid=False, updated=0)

    updated = 0
    async with database.transaction() as connection:
        for candidate, vector in zip(batch.candidates, batch.vectors, strict=True):
            applied = await connection.fetchval(
                PERSIST_SQL[batch.corpus],
                embedding_to_pgvector(list(vector)),
                BGE_M3_EMBEDDING_SPEC_VERSION,
                candidate.id,
                candidate.revision,
            )
            updated += applied is True
    return EmbeddingWriteResult(valid=True, updated=updated)


async def wake_embedding_worker(database: Database) -> None:
    """Send a wake-only notification whose payload is never durable work."""
    await database.execute(
        "SELECT pg_notify('embedding_dirty', $1)",
        EMBEDDING_WAKE_PAYLOAD,
    )


async def rebuild_embedding_lifecycle(
    database: Database,
) -> tuple[EmbeddingLifecycleCount, ...]:
    """Atomically dirty every BGE-M3 corpus row and wake the worker."""
    async with database.transaction() as connection:
        await rebuild_embedding_lifecycle_in_transaction(connection)
    return await lifecycle_counts(database)


async def rebuild_embedding_lifecycle_in_transaction(
    connection: PoolConnectionProxy[Record],
) -> None:
    """Invalidate all BGE-M3 corpora inside the caller-owned transaction."""
    for corpus in EmbeddingCorpus:
        await connection.execute(REBUILD_SQL[corpus])
    await connection.execute(
        "SELECT pg_notify('embedding_dirty', $1)",
        EMBEDDING_WAKE_PAYLOAD,
    )


def _candidate_from_row(corpus: EmbeddingCorpus, row: Record) -> EmbeddingCandidate:
    """Parse one trusted allowlisted query result into a captured lifecycle candidate."""
    match corpus:
        case EmbeddingCorpus.QUESTION:
            text = canonical_question_text(str(row["name"]), str(row["content"]))
        case EmbeddingCorpus.CHUNK:
            section = row["section"]
            text = canonical_chunk_text(
                str(row["document_title"]),
                str(section) if section is not None else None,
                str(row["content"]),
            )
        case EmbeddingCorpus.DIPLOMA:
            description = row["description"]
            text = canonical_diploma_text(
                str(row["title"]),
                str(description) if description is not None else None,
            )
        case EmbeddingCorpus.PROFESSOR_DOCUMENT:
            abstract = row["abstract"]
            text = canonical_professor_document_text(
                str(row["title"]),
                str(abstract) if abstract is not None else None,
            )
        case unreachable:
            assert_never(unreachable)
    return EmbeddingCandidate(
        corpus=corpus,
        id=row["id"],
        text=text,
        revision=int(row["embedding_revision"]),
    )
