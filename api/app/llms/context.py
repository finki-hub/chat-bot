import asyncio
import logging
import unicodedata
from dataclasses import dataclass

import httpx

from app.data.connection import Database
from app.data.documents import get_closest_chunks
from app.data.links import fetch_links_for_context
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

# Guarantee at least this many FAQ answers in the final context (if available), so a
# long document's many chunks cannot crowd FAQ out of the shared rerank pool.
RESERVED_FAQ_SLOTS: int = 3
_RERANKER_TIMEOUT = httpx.Timeout(timeout=30.0)
_RERANKER_MAX_RETRIES: int = 1


@dataclass(frozen=True)
class _Candidate:
    key: str  # dedup key, namespaced 'Q:'/'C:' so a chunk never collides with a FAQ
    source: str  # 'faq' | 'chunk'
    rerank_text: str  # the candidate text scored by the cross-encoder
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
    rerank_text = f"Наслов: {label}\nСодржина: {c.content}"
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

        # The reranker returns each candidate's original index, so we map straight back
        # to the candidate list — no fragile text matching that could silently drop a
        # result on any whitespace/serialization difference.
        ranked_candidates: list[_Candidate] = []
        dropped = 0
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
                dropped += 1
                continue
            ranked_candidates.append(candidates[idx])

        if dropped:
            logger.info(
                "Reranker dropped %d/%d candidates below RERANKER_MIN_SCORE=%.2f",
                dropped,
                len(ranked),
                settings.RERANKER_MIN_SCORE,
            )

        if not ranked_candidates and ranked:
            # Everything was scored but nothing cleared the floor — rather than return
            # empty context, keep the single best-scored candidate (ranked is desc).
            best_idx = ranked[0]["index"]
            if 0 <= best_idx < len(candidates):
                logger.warning(
                    "All %d reranked candidates were below RERANKER_MIN_SCORE=%.2f; "
                    "keeping the top-scored one",
                    len(ranked),
                    settings.RERANKER_MIN_SCORE,
                )
                ranked_candidates = [candidates[best_idx]]

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


# Bound the user-editable catalog's footprint in every prompt.
_LINKS_MAX_ROWS = 50
_LINK_NAME_MAX = 80
_LINK_URL_MAX = 2048
_LINK_DESC_MAX = 200


def _sanitize_inline(text: str, max_len: int) -> str:
    """Flatten a field to a bounded inline span, stripping newlines/control chars so a stored value can't fabricate prompt structure (fake headers or bullets)."""
    spaced = "".join(" " if ch.isspace() else ch for ch in text)
    cleaned = "".join(
        ch for ch in spaced if not unicodedata.category(ch).startswith("C")
    )
    collapsed = " ".join(cleaned.split())
    if len(collapsed) > max_len:
        collapsed = collapsed[:max_len].rstrip() + "…"
    return collapsed


async def get_links_context(db: Database) -> str:
    """Render the whole (capped) `link` catalog as a context block; per-row and fault-tolerant, so one bad row or a DB error degrades to "" instead of a broken answer."""
    try:
        rows = await fetch_links_for_context(db, _LINKS_MAX_ROWS)
    except Exception:
        logger.exception("Failed to load the link catalog for context")
        return ""

    lines: list[str] = []
    for row in rows:
        name = _sanitize_inline(row["name"] or "", _LINK_NAME_MAX)
        url = _sanitize_inline(row["url"] or "", _LINK_URL_MAX)
        if not name or not url:
            continue
        description = _sanitize_inline(row["description"] or "", _LINK_DESC_MAX)
        line = f"- {name}: {url}"
        if description:
            line += f" ({description})"
        lines.append(line)

    if not lines:
        return ""

    return "Корисни линкови:\n" + "\n".join(lines)
