import logging

from app.data.connection import Database
from app.data.diplomas import get_closest_diplomas
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.reranker import post_rerank
from app.llms.text_utils import _prepare_text_for_embedding
from app.recommenders.recommend import RetrievedDiploma
from app.recommenders.text import build_proposal_text
from app.schemas.diplomas import DiplomaSchema
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

# Short titles can score below the floor on every defense; keep a few so the
# planner still answers, marked for manual review.
_BELOW_FLOOR_FALLBACK_K = 3


def _to_retrieved(
    diploma: DiplomaSchema,
    rerank_score: float | None,
    *,
    below_rerank_floor: bool = False,
) -> RetrievedDiploma:
    distance = diploma.distance if diploma.distance is not None else 1.0
    return RetrievedDiploma(
        id=diploma.id,
        external_id=diploma.external_id,
        title=diploma.title,
        mentor=diploma.mentor,
        member1=diploma.member1,
        member2=diploma.member2,
        similarity=1.0 - distance,
        rerank_score=rerank_score,
        date_of_submission=diploma.date_of_submission,
        below_rerank_floor=below_rerank_floor,
    )


async def retrieve_similar_diplomas(
    db: Database,
    query_text: str,
    model: Model,
    initial_k: int,
    top_k: int,
    *,
    exclude_external_id: str | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievedDiploma]:
    embedded = (
        query_embedding
        if query_embedding is not None
        else await generate_embeddings(
            _prepare_text_for_embedding(query_text, model, is_document=False),
            model,
        )
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
        response = await post_rerank(
            {"query": query_text, "documents": rerank_texts},
        )
        ranked = response.json()["reranked_documents"]

        selected: list[RetrievedDiploma] = []
        below_floor: list[RetrievedDiploma] = []
        for item in ranked:
            idx = item["index"]
            if not 0 <= idx < len(candidates):
                logger.warning(
                    "Reranker returned out-of-range index %d (have %d candidates)",
                    idx,
                    len(candidates),
                )
                continue
            if item["score"] < settings.DIPLOMA_RERANKER_MIN_SCORE:
                if len(below_floor) < _BELOW_FLOOR_FALLBACK_K:
                    below_floor.append(
                        _to_retrieved(
                            candidates[idx],
                            item["score"],
                            below_rerank_floor=True,
                        ),
                    )
                continue
            selected.append(_to_retrieved(candidates[idx], item["score"]))
            if len(selected) >= top_k:
                break
    # ruff: ignore[BLE001] -- reranking degrades to the already ordered vector results
    except Exception as exc:
        logger.warning(
            "Diploma reranking failed; using vector order error_type=%s",
            type(exc).__name__,
        )
        return [_to_retrieved(d, None) for d in candidates[:top_k]]

    return selected or below_floor[:top_k]
