import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, assert_never

import httpx

from app.data.connection import Database
from app.data.embedding_lifecycle import (
    EmbeddingBatch,
    EmbeddingCandidate,
    fetch_dirty_embeddings,
    persist_embedding_batch,
)
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.llms.embedding_fills import EMBEDDING_BATCH_SIZE
from app.llms.embedding_generation import generate_embeddings
from app.llms.models import Model

logger = logging.getLogger(__name__)

type EmbeddingGenerator = Callable[[list[str], Model], Awaitable[list[list[float]]]]


@dataclass(frozen=True, slots=True)
class DirtyDrainReport:
    """The observable result of one complete durable dirty-work drain."""

    processed_batches: int
    updated_rows: int
    failed_batches: int
    invalid_batches: int


type BatchPreparation = (
    EmbeddingBatch
    | Literal[
        "provider_failure",
        "malformed_provider_output",
    ]
)


async def _prepare_embedding_batch(
    generate: EmbeddingGenerator,
    candidates: tuple[EmbeddingCandidate, ...],
    corpus: EmbeddingCorpus,
) -> BatchPreparation:
    try:
        vectors = await generate(
            [candidate.text for candidate in candidates],
            Model.BGE_M3_LOCAL,
        )
    except (OSError, ValueError, httpx.HTTPError) as error:
        logger.warning(
            "embedding_worker.drain provider_failed corpus=%s count=%d error_type=%s",
            corpus.value,
            len(candidates),
            type(error).__name__,
        )
        return "provider_failure"

    try:
        return EmbeddingBatch(
            corpus=corpus,
            candidates=candidates,
            vectors=tuple(tuple(vector) for vector in vectors),
        )
    except TypeError:
        logger.warning(
            "embedding_worker.drain provider_malformed corpus=%s count=%d",
            corpus.value,
            len(candidates),
        )
        return "malformed_provider_output"


async def _drain_corpus(
    database: Database,
    generate: EmbeddingGenerator,
    corpus: EmbeddingCorpus,
) -> DirtyDrainReport:
    processed_batches = 0
    updated_rows = 0
    failed_batches = 0
    invalid_batches = 0

    while candidates := await fetch_dirty_embeddings(
        database,
        corpus,
        EMBEDDING_BATCH_SIZE,
    ):
        processed_batches += 1
        preparation = await _prepare_embedding_batch(generate, candidates, corpus)
        match preparation:
            case EmbeddingBatch() as batch:
                result = await persist_embedding_batch(database, batch)
                if not result.valid:
                    invalid_batches += 1
                    logger.warning(
                        "embedding_worker.drain provider_malformed corpus=%s count=%d",
                        corpus.value,
                        len(candidates),
                    )
                    break
                updated_rows += result.updated
            case "provider_failure":
                failed_batches += 1
                break
            case "malformed_provider_output":
                invalid_batches += 1
                break
            case unreachable:
                assert_never(unreachable)

    return DirtyDrainReport(
        processed_batches,
        updated_rows,
        failed_batches,
        invalid_batches,
    )


async def drain_dirty_embeddings(
    database: Database,
    generate: EmbeddingGenerator = generate_embeddings,
    corpus: EmbeddingCorpus | None = None,
) -> DirtyDrainReport:
    """Drain every currently dirty BGE-M3 row without streaming or polling."""
    corpora = (corpus,) if corpus is not None else tuple(EmbeddingCorpus)
    reports = [
        await _drain_corpus(database, generate, current_corpus)
        for current_corpus in corpora
    ]

    report = DirtyDrainReport(
        processed_batches=sum(item.processed_batches for item in reports),
        updated_rows=sum(item.updated_rows for item in reports),
        failed_batches=sum(item.failed_batches for item in reports),
        invalid_batches=sum(item.invalid_batches for item in reports),
    )
    logger.info(
        "embedding_worker.drain completed batches=%d updated=%d",
        report.processed_batches,
        report.updated_rows,
    )
    return report
