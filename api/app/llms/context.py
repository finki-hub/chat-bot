import asyncio
import logging
from dataclasses import dataclass

import httpx

from app.data.connection import Database
from app.data.documents import get_closest_chunks
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.prompts import HYDE_SYSTEM_PROMPT
from app.llms.query_transform import transform_query
from app.llms.text_utils import _prepare_text_for_embedding
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.http_client import get_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

RERANKER_MIN_SCORE: float = 0.1
# Guarantee at least this many FAQ answers in the final context (if available), so a
# long document's many chunks cannot crowd FAQ out of the shared rerank pool.
RESERVED_FAQ_SLOTS: int = 3
_RERANKER_TIMEOUT = httpx.Timeout(timeout=30.0)
_RERANKER_MAX_RETRIES: int = 1


@dataclass(frozen=True)
class _Candidate:
    key: str  # dedup key, namespaced 'Q:'/'C:' so a chunk never collides with a FAQ
    source: str  # 'faq' | 'chunk'
    rerank_text: str  # sent to the cross-encoder; MUST be unique per candidate
    context_text: str  # rendered into the final context string


async def _post_rerank(payload: dict) -> httpx.Response:
    client = get_http_client()
    for attempt in range(_RERANKER_MAX_RETRIES + 1):
        try:
            response = await client.post(
                f"{settings.GPU_API_URL}/rerank/",
                json=payload,
                timeout=_RERANKER_TIMEOUT,
            )
            response.raise_for_status()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            if attempt < _RERANKER_MAX_RETRIES:
                logger.warning(
                    "Reranker attempt %d failed (%s), retrying...",
                    attempt + 1,
                    exc,
                )
                continue
            raise
        else:
            return response
    raise RuntimeError(
        "Unreachable: reranker retry loop exited without return or raise",
    )


async def _embed_variant(
    text: str,
    embedding_model: Model,
    *,
    is_document: bool,
) -> list[float]:
    prepared = _prepare_text_for_embedding(
        text,
        embedding_model,
        is_document=is_document,
    )
    return await generate_embeddings(prepared, embedding_model)


async def _search_both(
    db: Database,
    embedding: list[float],
    embedding_model: Model,
    limit: int,
) -> tuple[list[QuestionSchema], list[ChunkSchema]]:
    questions, chunks = await asyncio.gather(
        get_closest_questions(db, embedding, embedding_model, limit=limit),
        get_closest_chunks(db, embedding, embedding_model, limit=limit),
    )
    return questions, chunks


def _question_candidate(q: QuestionSchema) -> _Candidate:
    rerank_text = f"Наслов: {q.name}\nСодржина: {q.content}"
    sources = (
        "Извори: " + ", ".join(f"{k}: {v}" for k, v in q.links.items())
        if q.links
        else None
    )
    context_text = "\n".join(
        part
        for part in [f"Наслов: {q.name}", f"Содржина: {q.content}", sources]
        if part is not None
    )
    return _Candidate(
        key=f"Q:{q.id}",
        source="faq",
        rerank_text=rerank_text,
        context_text=context_text,
    )


def _chunk_candidate(c: ChunkSchema) -> _Candidate:
    label = f"{c.document_title} ({c.section})" if c.section else c.document_title
    # Include document_id#chunk_index so the rerank_text is unique even when two chunks
    # share identical visible text — the cross-encoder match-back relies on string equality.
    rerank_text = (
        f"Наслов: {label} [{c.document_id}#{c.chunk_index}]\nСодржина: {c.content}"
    )
    context_text = f"Извор: {label}\nСодржина: {c.content}"
    return _Candidate(
        key=f"C:{c.id}",
        source="chunk",
        rerank_text=rerank_text,
        context_text=context_text,
    )


def _build_candidates(
    question_lists: list[list[QuestionSchema]],
    chunk_lists: list[list[ChunkSchema]],
) -> list[_Candidate]:
    seen: set[str] = set()
    merged: list[_Candidate] = []
    for questions, chunks in zip(question_lists, chunk_lists, strict=True):
        for cand in (
            *map(_question_candidate, questions),
            *map(_chunk_candidate, chunks),
        ):
            if cand.key not in seen:
                seen.add(cand.key)
                merged.append(cand)
    return merged


def _select_with_faq_reservation(
    ranked: list[_Candidate],
    top_k: int,
) -> list[_Candidate]:
    """Take top_k ranked, but guarantee up to RESERVED_FAQ_SLOTS FAQ entries by displacing the lowest-ranked chunks."""
    primary = ranked[:top_k]
    faqs = [c for c in ranked if c.source == "faq"]
    reserve = min(RESERVED_FAQ_SLOTS, len(faqs))
    in_primary = sum(1 for c in primary if c.source == "faq")
    if in_primary >= reserve:
        return primary

    primary_keys = {c.key for c in primary}
    extra = [c for c in faqs if c.key not in primary_keys][: reserve - in_primary]
    if not extra:
        return primary

    result = list(primary)
    needed = len(extra)
    for i in range(len(result) - 1, -1, -1):
        if needed == 0:
            break
        if result[i].source == "chunk":
            result.pop(i)
            needed -= 1
    result.extend(extra)

    order = {c.key: i for i, c in enumerate(ranked)}
    result.sort(key=lambda c: order[c.key])
    return result[:top_k]


async def get_retrieved_context(
    db: Database,
    query: str,
    embedding_model: Model,
    query_transform_model: Model,
    *,
    initial_k: int = 30,
    top_k: int = 10,
) -> str:
    """Multi-query (original + rewritten + HyDE) retrieval over FAQ and chunks, reranked by a cross-encoder with vector-order fallback."""

    logger.info(
        "Retrieving context for query: '%s' with embedding model: %s",
        query,
        embedding_model,
    )

    try:
        rewritten_query, hyde_passage = await asyncio.gather(
            transform_query(
                query,
                query_transform_model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=128,
            ),
            transform_query(
                query,
                query_transform_model,
                system_prompt=HYDE_SYSTEM_PROMPT,
                temperature=0.7,
                top_p=0.9,
                max_tokens=200,
            ),
        )
    except Exception as e:
        raise RetrievalError("Failed during query transform / HyDE generation") from e

    logger.info("Transformed query: '%s'", rewritten_query)
    logger.info("HyDE passage: '%s'", hyde_passage)

    per_query_k = initial_k // 3 + 1

    try:
        emb_original, emb_rewritten, emb_hyde = await asyncio.gather(
            _embed_variant(query, embedding_model, is_document=False),
            _embed_variant(rewritten_query, embedding_model, is_document=False),
            _embed_variant(hyde_passage, embedding_model, is_document=True),
        )

        hyde_res, rewritten_res, original_res = await asyncio.gather(
            _search_both(db, emb_hyde, embedding_model, per_query_k),
            _search_both(db, emb_rewritten, embedding_model, per_query_k),
            _search_both(db, emb_original, embedding_model, per_query_k),
        )

        candidates = _build_candidates(
            [hyde_res[0], rewritten_res[0], original_res[0]],
            [hyde_res[1], rewritten_res[1], original_res[1]],
        )

        logger.info(
            "Multi-query candidates: %d (faq=%d, chunk=%d)",
            len(candidates),
            sum(1 for c in candidates if c.source == "faq"),
            sum(1 for c in candidates if c.source == "chunk"),
        )

        if not candidates:
            return ""

    except Exception as e:
        raise RetrievalError("Failed during multi-query vector search") from e

    rerank_texts = [c.rerank_text for c in candidates]

    try:
        logger.info("Sending %d candidates to re-ranker...", len(rerank_texts))

        response = await _post_rerank(
            {"query": rewritten_query, "documents": rerank_texts},
        )
        ranked = response.json()["reranked_documents"]

        # Build a mutable list of (index, text) so duplicate documents each map to their
        # own original position instead of colliding in a dict.
        available = list(enumerate(rerank_texts))

        ranked_candidates: list[_Candidate] = []
        for item in ranked:
            if item["score"] < RERANKER_MIN_SCORE:
                continue
            doc_text = str(item["document"])
            for pos, (orig_idx, candidate_text) in enumerate(available):
                if candidate_text == doc_text:
                    ranked_candidates.append(candidates[orig_idx])
                    available.pop(pos)
                    break

        final = _select_with_faq_reservation(ranked_candidates, top_k)

        logger.info(
            "Selected %d documents after reranking (faq=%d, chunk=%d)",
            len(final),
            sum(1 for c in final if c.source == "faq"),
            sum(1 for c in final if c.source == "chunk"),
        )
    except Exception:
        logger.exception(
            "Reranking call failed. Using vector search order as a fallback",
        )
        final = _select_with_faq_reservation(candidates, top_k)

    return "\n\n---\n\n".join(c.context_text for c in final)
