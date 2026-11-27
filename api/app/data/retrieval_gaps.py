from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from app.data.connection import Database
from app.llms.agents import SearchAttempt

RetrievalGapReason = Literal["no_results", "weak_results"]


@dataclass(frozen=True, slots=True)
class RetrievalGap:
    response_id: UUID
    interface: Literal["discord", "web"]
    reason: RetrievalGapReason
    search_queries: list[str | None]
    inference_model: str
    embeddings_model: str
    reranker_score_max: float | None
    outcome: str


def retrieval_gap_reason(
    searches: Sequence[SearchAttempt],
) -> RetrievalGapReason | None:
    """Why an answer's searches found nothing usable; None when one found a match
    above the reranker floor, or when nothing was searched. A search whose reranker
    failed counts as a match: that is an outage, not a missing topic."""
    if not searches:
        return None
    if not any(search.hit for search in searches):
        return "no_results"
    if all(search.weak or not search.hit for search in searches):
        return "weak_results"
    return None


async def delete_orphaned_retrieval_gaps(db: Database) -> None:
    """Drop rows whose answer was never stored (suppressed Discord storage, a
    deletion racing the write) or was replaced by a regeneration or an edit."""
    await db.execute(
        """
        DELETE FROM retrieval_gap AS gap
        WHERE gap.created_at < NOW() - INTERVAL '1 hour'
          AND NOT EXISTS (
              SELECT 1 FROM chat_message AS message
              WHERE message.response_id = gap.response_id
                AND message.role = 'assistant'
          )
        """,
    )


async def insert_retrieval_gap(db: Database, gap: RetrievalGap) -> None:
    await db.execute(
        """
        INSERT INTO retrieval_gap (
            response_id, interface, reason, search_queries, inference_model,
            embeddings_model, reranker_score_max, outcome
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (response_id) DO NOTHING
        """,
        gap.response_id,
        gap.interface,
        gap.reason,
        gap.search_queries,
        gap.inference_model,
        gap.embeddings_model,
        gap.reranker_score_max,
        gap.outcome,
    )
