import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from app.data.connection import Database
from app.data.embedding_lifecycle import (
    EmbeddingBatch,
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


async def drain_dirty_embeddings(
    database: Database,
    generate: EmbeddingGenerator = generate_embeddings,
    corpus: EmbeddingCorpus | None = None,
) -> DirtyDrainReport:
    """Drain every currently dirty BGE-M3 row without streaming or polling."""
    processed_batches = 0
    updated_rows = 0
    failed_batches = 0
    invalid_batches = 0

    corpora = (corpus,) if corpus is not None else tuple(EmbeddingCorpus)
    for current_corpus in corpora:
        while True:
            candidates = await fetch_dirty_embeddings(
                database,
                current_corpus,
                EMBEDDING_BATCH_SIZE,
            )
            if not candidates:
                break

            processed_batches += 1
            try:
                vectors = await generate(
                    [candidate.text for candidate in candidates],
                    Model.BGE_M3_LOCAL,
                )
            except (OSError, ValueError, httpx.HTTPError) as error:
                failed_batches += 1
                logger.warning(
                    "embedding_worker.drain provider_failed corpus=%s count=%d error_type=%s",
                    current_corpus.value,
                    len(candidates),
                    type(error).__name__,
                )
                break

            try:
                batch = EmbeddingBatch(
                    corpus=current_corpus,
                    candidates=candidates,
                    vectors=tuple(tuple(vector) for vector in vectors),
                )
            except TypeError:
                invalid_batches += 1
                logger.warning(
                    "embedding_worker.drain provider_malformed corpus=%s count=%d",
                    current_corpus.value,
                    len(candidates),
                )
                break

            result = await persist_embedding_batch(database, batch)
            if not result.valid:
                invalid_batches += 1
                logger.warning(
                    "embedding_worker.drain provider_malformed corpus=%s count=%d",
                    current_corpus.value,
                    len(candidates),
                )
                break
            updated_rows += result.updated

    report = DirtyDrainReport(
        processed_batches,
        updated_rows,
        failed_batches,
        invalid_batches,
    )
    logger.info(
        "embedding_worker.drain completed batches=%d updated=%d",
        report.processed_batches,
        report.updated_rows,
    )
    return report
