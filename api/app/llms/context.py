import asyncio
import logging

import httpx

from app.data.connection import Database
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.prompts import HYDE_SYSTEM_PROMPT
from app.llms.query_transform import transform_query
from app.llms.text_utils import _prepare_text_for_embedding
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.http_client import get_http_client
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

RERANKER_MIN_SCORE: float = 0.1
_RERANKER_TIMEOUT = httpx.Timeout(timeout=30.0)
_RERANKER_MAX_RETRIES: int = 1


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
            return response
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            if attempt < _RERANKER_MAX_RETRIES:
                logger.warning(
                    "Reranker attempt %d failed (%s), retrying...",
                    attempt + 1,
                    exc,
                )
                continue
            raise
    raise RuntimeError("Unreachable: reranker retry loop exited without return or raise")


async def _embed_and_search(
    db: Database,
    text: str,
    embedding_model: Model,
    limit: int,
    *,
    is_document: bool = False,
) -> list[QuestionSchema]:
    prepared = _prepare_text_for_embedding(text, embedding_model, is_document=is_document)
    embedding = await generate_embeddings(prepared, embedding_model)
    return await get_closest_questions(db, embedding, embedding_model, limit=limit)


async def _merge_candidates(
    candidate_lists: list[list[QuestionSchema]],
) -> list[QuestionSchema]:
    seen: set[str] = set()
    merged: list[QuestionSchema] = []
    for candidates in candidate_lists:
        for q in candidates:
            qid = str(q.id)
            if qid not in seen:
                seen.add(qid)
                merged.append(q)
    return merged


async def get_retrieved_context(
    db: Database,
    query: str,
    embedding_model: Model,
    query_transform_model: Model,
    *,
    initial_k: int = 30,
    top_k: int = 10,
) -> str:
    """
    Multi-query retrieval: generates a rewritten query and HyDE passage,
    embeds all three variants (original, rewritten, HyDE) in parallel,
    unions the candidate sets, and reranks with a cross-encoder.
    Falls back to vector order on reranker failure.
    """

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
        original_results, rewritten_results, hyde_results = await asyncio.gather(
            _embed_and_search(db, query, embedding_model, per_query_k),
            _embed_and_search(db, rewritten_query, embedding_model, per_query_k),
            _embed_and_search(
                db, hyde_passage, embedding_model, per_query_k, is_document=True,
            ),
        )

        initial_candidates = await _merge_candidates(
            [hyde_results, rewritten_results, original_results],
        )

        logger.info(
            "Multi-query candidates: %d (hyde=%d, rewritten=%d, original=%d)",
            len(initial_candidates),
            len(hyde_results),
            len(rewritten_results),
            len(original_results),
        )

        if not initial_candidates:
            return ""

    except Exception as e:
        raise RetrievalError("Failed during multi-query vector search") from e

    rerank_docs: list[str] = []
    context_docs: list[str] = []

    for q in initial_candidates:
        title_content = f"Наслов: {q.name}\nСодржина: {q.content}"
        rerank_docs.append(title_content)

        sources = (
            "Извори: " + ", ".join(f"{k}: {v}" for k, v in q.links.items())
            if q.links
            else None
        )
        full_doc = "\n".join(
            part
            for part in [f"Наслов: {q.name}", f"Содржина: {q.content}", sources]
            if part is not None
        )
        context_docs.append(full_doc)

    try:
        logger.info("Sending %d candidates to re-ranker...", len(rerank_docs))

        rerank_payload = {
            "query": rewritten_query,
            "documents": rerank_docs,
        }

        response = await _post_rerank(rerank_payload)
        ranked = response.json()["reranked_documents"]

        # Build a mutable list of (index, text) so duplicate documents each
        # map to their own original position instead of colliding in a dict.
        available = list(enumerate(rerank_docs))

        final_docs: list[str] = []
        for item in ranked:
            if item["score"] < RERANKER_MIN_SCORE:
                continue
            doc_text = str(item["document"])
            for pos, (orig_idx, candidate) in enumerate(available):
                if candidate == doc_text:
                    final_docs.append(context_docs[orig_idx])
                    available.pop(pos)
                    break
            if len(final_docs) >= top_k:
                break

        logger.info(
            "Selected %d documents after reranking and score filtering",
            len(final_docs),
        )
    except Exception:
        logger.exception(
            "Reranking call failed. Using vector search order as a fallback",
        )
        final_docs = context_docs[:top_k]

    return "\n\n---\n\n".join(final_docs)
