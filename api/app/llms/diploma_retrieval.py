"""Diploma read-path retrieval: embed proposal title, KNN over diploma corpus, rerank, return RetrievedDiploma candidates."""

import logging

from app.data.connection import Database
from app.data.diplomas import get_closest_diplomas
from app.llms.context import _post_rerank
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.text_utils import _prepare_text_for_embedding
from app.recommenders.recommend import RetrievedDiploma
from app.recommenders.text import build_proposal_text
from app.schemas.diplomas import DiplomaSchema
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()


def _to_retrieved(
    diploma: DiplomaSchema,
    rerank_score: float | None,
) -> RetrievedDiploma:
    distance = diploma.distance if diploma.distance is not None else 1.0
    return RetrievedDiploma(
        id=diploma.id,
        external_id=diploma.external_id,
        mentor=diploma.mentor,
        member1=diploma.member1,
        member2=diploma.member2,
        similarity=1.0 - distance,
        rerank_score=rerank_score,
        date_of_submission=diploma.date_of_submission,
    )


async def retrieve_similar_diplomas(
    db: Database,
    query_text: str,
    model: Model,
    initial_k: int,
    top_k: int,
    *,
    exclude_external_id: str | None = None,
) -> list[RetrievedDiploma]:
    """Retrieve the top_k historical defenses most similar to the proposal title.

    The query is the bare title (no description; E5 query: prefix). After KNN we rerank
    with the cross-encoder, keep candidates scoring >= RERANKER_MIN_SCORE, and take top_k.
    On any rerank failure we fall back to vector order (first top_k), mirroring
    get_retrieved_context.
    """
    embedded = await generate_embeddings(
        _prepare_text_for_embedding(query_text, model, is_document=False),
        model,
    )

    candidates = await get_closest_diplomas(
        db,
        embedded,
        model,
        limit=initial_k,
        exclude_external_id=exclude_external_id,
    )

    if not candidates:
        return []

    rerank_texts = [build_proposal_text(d.title, d.description) for d in candidates]

    try:
        response = await _post_rerank(
            {"query": query_text, "documents": rerank_texts},
        )
        ranked = response.json()["reranked_documents"]

        selected: list[RetrievedDiploma] = []
        for item in ranked:
            idx = item["index"]
            if not 0 <= idx < len(candidates):
                logger.warning(
                    "Reranker returned out-of-range index %d (have %d candidates)",
                    idx,
                    len(candidates),
                )
                continue
            if item["score"] < settings.RERANKER_MIN_SCORE:
                continue
            selected.append(_to_retrieved(candidates[idx], item["score"]))
            if len(selected) >= top_k:
                break
    except Exception:
        logger.exception(
            "Diploma reranking call failed. Using vector search order as a fallback",
        )
        return [_to_retrieved(d, None) for d in candidates[:top_k]]

    return selected
