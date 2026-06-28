import asyncio
import logging
import unicodedata
from dataclasses import dataclass
from uuid import UUID

import httpx

from app.data.connection import Database
from app.data.documents import get_chunks_window, get_closest_chunks
from app.data.links import fetch_links_for_context
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.prompts import CONTEXTUALIZE_SYSTEM_PROMPT, HYDE_SYSTEM_PROMPT
from app.llms.query_transform import transform_query
from app.llms.text_utils import _prepare_text_for_embedding
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.http_client import get_http_client
from app.utils.settings import Settings
from app.utils.timing import (
    current_distinct_id,
    current_response_id,
    record_reranker_scores,
    record_retrieval_ids,
    record_retrieval_shape,
    timed,
)

logger = logging.getLogger(__name__)

settings = Settings()

# Guarantee at least this many FAQ answers in the final context (if available), so a
# long document's many chunks cannot crowd FAQ out of the shared rerank pool.
RESERVED_FAQ_SLOTS: int = 3
_RERANKER_TIMEOUT = httpx.Timeout(timeout=30.0)
_RERANKER_MAX_RETRIES: int = 1
# Chunks pulled in on each side of a retrieved chunk, to give the model a contiguous passage.
_NEIGHBOR_WINDOW: int = 1
_RETRIEVAL_IDS_CAP: int = 10


@dataclass(frozen=True)
class _Candidate:
    key: str  # dedup key, namespaced 'Q:'/'C:' so a chunk never collides with a FAQ
    source: str  # 'faq' | 'chunk'
    rerank_text: str  # the candidate text scored by the cross-encoder
    context_text: str  # rendered into the final context string
    distance: float | None = None  # carried only for the DEBUG trace
    doc_id: UUID | None = None  # chunk identity for neighbor expansion (None for FAQ)
    chunk_index: int | None = None


async def _post_rerank(payload: dict) -> httpx.Response:
    client = get_http_client()
    for attempt in range(_RERANKER_MAX_RETRIES + 1):
        try:
            headers: dict[str, str] = {}
            rid = current_response_id()
            if rid:
                headers["X-Response-Id"] = rid
            did = current_distinct_id()
            if did:
                headers["X-Distinct-Id"] = did
            response = await client.post(
                f"{settings.GPU_API_URL}/rerank/",
                json=payload,
                headers=headers or None,
                timeout=_RERANKER_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
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
    return await generate_embeddings(prepared, embedding_model, is_document=is_document)


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
        distance=q.distance,
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
        distance=c.distance,
        doc_id=c.document_id,
        chunk_index=c.chunk_index,
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
    displaced: list[str] = []
    for i in range(len(result) - 1, -1, -1):
        if needed == 0:
            break
        if result[i].source == "chunk":
            displaced.append(result[i].key)
            result.pop(i)
            needed -= 1
    result.extend(extra)

    if displaced and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "FAQ reservation displaced %s to insert FAQs %s",
            displaced,
            [c.key for c in extra],
        )

    order = {c.key: i for i, c in enumerate(ranked)}
    result.sort(key=lambda c: order[c.key])
    return result[:top_k]


async def get_retrieved_context(
    db: Database,
    query: str,
    embedding_model: Model,
    query_transform_model: Model,
    *,
    history_text: str | None = None,
    initial_k: int = 30,
    top_k: int = 10,
) -> str:
    """Multi-query (original + rewritten + HyDE) retrieval over FAQ and chunks, reranked by a cross-encoder with vector-order fallback."""

    logger.info(
        "Retrieving context for query: '%s' with embedding model: %s",
        query,
        embedding_model,
    )

    with timed("retrieval.contextualize"):
        search_query = await _contextualize_query(
            query,
            query_transform_model,
            history_text,
        )

    try:
        with timed("retrieval.rewrite_hyde"):
            rewritten_query, hyde_passage = await asyncio.gather(
                transform_query(
                    search_query,
                    query_transform_model,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=128,
                ),
                transform_query(
                    search_query,
                    query_transform_model,
                    system_prompt=HYDE_SYSTEM_PROMPT,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=200,
                ),
            )
    except Exception as e:
        raise RetrievalError("Failed during query transform / HyDE generation") from e

    # transform_query can return an empty/whitespace string on a refusal; a blank
    # rewrite makes the cross-encoder score every candidate 0.0 and bypass reranking.
    rewritten_query = rewritten_query.strip() or search_query
    hyde_passage = hyde_passage.strip() or search_query

    logger.info("Transformed query: '%s'", rewritten_query)
    logger.info("HyDE passage: '%s'", hyde_passage)

    per_query_k = initial_k // 3 + 1

    try:
        with timed("retrieval.embed"):
            emb_original, emb_rewritten, emb_hyde = await asyncio.gather(
                _embed_variant(search_query, embedding_model, is_document=False),
                _embed_variant(rewritten_query, embedding_model, is_document=False),
                _embed_variant(hyde_passage, embedding_model, is_document=True),
            )

        with timed("retrieval.vector_search"):
            hyde_res, rewritten_res, original_res = await asyncio.gather(
                _search_both(db, emb_hyde, embedding_model, per_query_k),
                _search_both(db, emb_rewritten, embedding_model, per_query_k),
                _search_both(db, emb_original, embedding_model, per_query_k),
            )

        candidates = _build_candidates(
            [hyde_res[0], rewritten_res[0], original_res[0]],
            [hyde_res[1], rewritten_res[1], original_res[1]],
        )

        distances = [c.distance for c in candidates if c.distance is not None]
        record_retrieval_shape(
            len(candidates),
            min(distances) if distances else None,
        )

        logger.info(
            "Multi-query candidates: %d (faq=%d, chunk=%d)",
            len(candidates),
            sum(1 for c in candidates if c.source == "faq"),
            sum(1 for c in candidates if c.source == "chunk"),
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Retrieved candidates (key, source, distance): %s",
                [(c.key, c.source, c.distance) for c in candidates],
            )

        if not candidates:
            return ""

    except Exception as e:
        raise RetrievalError("Failed during multi-query vector search") from e

    rerank_texts = [c.rerank_text for c in candidates]

    try:
        logger.info("Sending %d candidates to re-ranker...", len(rerank_texts))

        with timed("retrieval.rerank"):
            response = await _post_rerank(
                {"query": rewritten_query, "documents": rerank_texts},
            )
        ranked = response.json()["reranked_documents"]

        # The reranker returns each candidate's original index, so we map straight back
        # to the candidate list — no fragile text matching that could silently drop a
        # result on any whitespace/serialization difference.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Rerank scores (key, score): %s",
                [
                    (candidates[item["index"]].key, round(item["score"], 4))
                    for item in ranked
                    if 0 <= item["index"] < len(candidates)
                ],
            )

        ranked_candidates: list[_Candidate] = []
        dropped: list[tuple[str, str, float]] = []
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
                dropped.append(
                    (candidates[idx].key, candidates[idx].source, item["score"]),
                )
                continue
            ranked_candidates.append(candidates[idx])

        record_reranker_scores(
            [item["score"] for item in ranked if 0 <= item["index"] < len(candidates)],
            above_threshold=len(ranked_candidates),
        )

        if dropped:
            logger.info(
                "Reranker dropped %d/%d candidates below RERANKER_MIN_SCORE=%.2f",
                len(dropped),
                len(ranked),
                settings.RERANKER_MIN_SCORE,
            )
            logger.debug(
                "Dropped candidates (key, source, score): %s",
                [(key, source, round(score, 4)) for key, source, score in dropped],
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

    record_retrieval_ids([c.key for c in final[:_RETRIEVAL_IDS_CAP]])

    with timed("retrieval.expand"):
        return await _expand_and_render(db, final)


async def _contextualize_query(
    query: str,
    query_transform_model: Model,
    history_text: str | None,
) -> str:
    """Fold prior turns into a standalone retrieval query, so a follow-up like
    'колку чини тоа?' retrieves on the full question rather than a context-free fragment.
    Returns the raw query unchanged when there's no history or the rewrite fails.
    """
    if not history_text:
        return query
    try:
        condensed = await transform_query(
            f"Претходен разговор:\n{history_text}\n\nНово прашање: {query}",
            query_transform_model,
            system_prompt=CONTEXTUALIZE_SYSTEM_PROMPT,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )
    except Exception:
        logger.exception("Query contextualization failed; using the raw query")
        return query
    condensed = condensed.strip()
    if condensed and condensed != query:
        logger.info("Contextualized query: '%s' -> '%s'", query, condensed)
        return condensed
    return query


async def _expand_and_render(db: Database, final: list[_Candidate]) -> str:
    """Render the final candidates, stitching each retrieved chunk together with its
    immediate neighbors into a single contiguous passage so an answer that spans a chunk
    boundary reads as one block. Falls back to plain per-candidate rendering on error.
    """
    refs: list[tuple[UUID, int]] = [
        (c.doc_id, c.chunk_index)
        for c in final
        if c.source == "chunk" and c.doc_id is not None and c.chunk_index is not None
    ]

    window_map: dict[tuple[UUID, int], ChunkSchema] = {}
    if refs:
        try:
            for ch in await get_chunks_window(db, refs, window=_NEIGHBOR_WINDOW):
                window_map[(ch.document_id, ch.chunk_index)] = ch
        except Exception:
            logger.exception("Neighbor expansion failed; rendering chunks unexpanded")

    return _render_blocks(final, window_map)


def _contiguous_runs(indices: list[int]) -> list[list[int]]:
    """Split a sorted list of chunk indices into maximal consecutive runs."""
    runs: list[list[int]] = []
    for idx in indices:
        if runs and idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def _render_passage(chunks: list[ChunkSchema]) -> str:
    """Render a contiguous run of chunks (ordered by chunk_index) as one source block."""
    title = chunks[0].document_title
    sections = {c.section for c in chunks}
    label = (
        f"{title} ({chunks[0].section})"
        if len(sections) == 1 and chunks[0].section
        else title
    )
    body = "\n".join(c.content for c in chunks)
    return f"Извор: {label}\nСодржина: {body}"


def _render_blocks(
    final: list[_Candidate],
    window_map: dict[tuple[UUID, int], ChunkSchema],
) -> str:
    """Stitch each retrieved chunk together with its neighbors into a contiguous,
    chunk-ordered passage; FAQ entries (and chunks whose window couldn't be fetched)
    render unchanged. Each block keeps the rerank position of its best-ranked chunk.
    Neighbors are unscored padding by design — acceptable at a ±1 window.
    """
    if not window_map:
        return "\n\n---\n\n".join(c.context_text for c in final)

    chunk_centers: dict[tuple[UUID, int], int] = {}
    items: list[tuple[int, str]] = []
    for rank, c in enumerate(final):
        if c.source == "chunk" and c.doc_id is not None and c.chunk_index is not None:
            ref = (c.doc_id, c.chunk_index)
            if ref in window_map:
                chunk_centers.setdefault(ref, rank)
                continue
        # FAQ, or a chunk whose window refetch came back short: never drop it.
        items.append((rank, c.context_text))

    included: dict[UUID, set[int]] = {}
    for doc_id, idx in chunk_centers:
        for delta in range(-_NEIGHBOR_WINDOW, _NEIGHBOR_WINDOW + 1):
            if (doc_id, idx + delta) in window_map:
                included.setdefault(doc_id, set()).add(idx + delta)

    for doc_id, indices in included.items():
        for run in _contiguous_runs(sorted(indices)):
            run_rank = min(
                chunk_centers[(doc_id, i)] for i in run if (doc_id, i) in chunk_centers
            )
            passage = _render_passage([window_map[(doc_id, i)] for i in run])
            items.append((run_rank, passage))

    items.sort(key=lambda item: item[0])
    return "\n\n---\n\n".join(text for _, text in items)


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
        with timed("links"):
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
