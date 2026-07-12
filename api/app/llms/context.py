import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import assert_never
from uuid import UUID

from app.data.connection import Database
from app.data.documents import get_chunks_window, get_closest_chunks
from app.data.questions import get_closest_questions
from app.llms.embeddings import generate_embeddings
from app.llms.models import ChatModel, Model
from app.llms.prompts import CONTEXTUALIZE_SYSTEM_PROMPT
from app.llms.provider_credentials import (
    LlmProviderCredentials,
    has_provider_credential,
)
from app.llms.query_modes import QueryTransformMode
from app.llms.query_transform import transform_query
from app.llms.query_variants import (
    QueryVariantKind,
    build_query_variants,
)
from app.llms.reranker import post_rerank as _post_rerank
from app.llms.retrieval_budget import retrieval_budget
from app.llms.retrieval_result import (
    RetrievalSource,
    RetrievalSourceLink,
    RetrievedContext,
    visible_sources,
)
from app.llms.text_utils import _prepare_text_for_embedding
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from app.utils.exceptions import RetrievalError
from app.utils.settings import Settings
from app.utils.timing import (
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
# Chunks pulled in on each side of a retrieved chunk, to give the model a contiguous passage.
_NEIGHBOR_WINDOW: int = 1
_RETRIEVAL_IDS_CAP: int = 10


@dataclass(frozen=True)
class _Candidate:
    key: str  # dedup key, namespaced 'Q:'/'C:' so a chunk never collides with a FAQ
    source: str  # 'faq' | 'chunk'
    rerank_text: str  # the candidate text scored by the cross-encoder
    context_text: str  # rendered into the final context string
    retrieval_source: RetrievalSource
    distance: float | None = None  # carried only for the DEBUG trace
    doc_id: UUID | None = None  # chunk identity for neighbor expansion (None for FAQ)
    chunk_index: int | None = None


async def _embed_variant(
    text: str,
    embedding_model: Model,
    *,
    is_document: bool,
    credentials: LlmProviderCredentials | None = None,
) -> list[float]:
    prepared = _prepare_text_for_embedding(
        text,
        embedding_model,
        is_document=is_document,
    )
    return await generate_embeddings(
        prepared,
        embedding_model,
        is_document=is_document,
        credentials=credentials,
    )


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


def _embedding_for_variant(
    kind: QueryVariantKind,
    text: str,
    embedding_model: Model,
    *,
    is_document: bool,
    original_embedding_task: asyncio.Task[list[float]],
    credentials: LlmProviderCredentials | None,
) -> Awaitable[list[float]]:
    match kind:
        case "raw":
            return original_embedding_task
        case "rewrite" | "hyde":
            return _embed_variant(
                text,
                embedding_model,
                is_document=is_document,
                credentials=credentials,
            )
        case unreachable:
            assert_never(unreachable)
    raise AssertionError(f"Unhandled query variant kind: {kind}")


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
        retrieval_source=RetrievalSource(
            id=str(q.id),
            kind="faq",
            title=q.name,
            links=tuple(
                RetrievalSourceLink(label=label, url=str(url))
                for label, url in (q.links or {}).items()
            ),
            snippet=q.content,
        ),
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
        retrieval_source=RetrievalSource(
            id=str(c.id),
            kind="chunk",
            title=c.document_title,
            chunk_index=c.chunk_index,
            section=c.section,
            snippet=c.content,
        ),
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
    query_transform_model: ChatModel,
    *,
    query_transform_mode: QueryTransformMode = QueryTransformMode.REWRITE_HYDE,
    history_text: str | None = None,
    initial_k: int = 30,
    top_k: int = 10,
    on_stage: Callable[[str], None] | None = None,
    credentials: LlmProviderCredentials | None = None,
) -> str:
    result = await get_retrieved_context_with_sources(
        db=db,
        query=query,
        embedding_model=embedding_model,
        query_transform_model=query_transform_model,
        query_transform_mode=query_transform_mode,
        history_text=history_text,
        initial_k=initial_k,
        top_k=top_k,
        on_stage=on_stage,
        credentials=credentials,
    )
    return result.text


async def get_retrieved_context_with_sources(
    db: Database,
    query: str,
    embedding_model: Model,
    query_transform_model: ChatModel,
    *,
    query_transform_mode: QueryTransformMode = QueryTransformMode.REWRITE_HYDE,
    history_text: str | None = None,
    initial_k: int = 30,
    top_k: int = 10,
    on_stage: Callable[[str], None] | None = None,
    credentials: LlmProviderCredentials | None = None,
) -> RetrievedContext:
    """Multi-query (original + rewritten + HyDE) retrieval over FAQ and chunks, reranked by a cross-encoder with vector-order fallback."""

    logger.info(
        "Retrieving context: query_len=%d embedding_model=%s",
        len(query),
        embedding_model.value,
    )

    def _stage(stage: str) -> None:
        if on_stage is not None:
            on_stage(stage)

    transform_available = has_provider_credential(credentials, query_transform_model)
    effective_transform_mode = (
        query_transform_mode if transform_available else QueryTransformMode.RAW
    )
    _stage("contextualize")

    with timed("retrieval.contextualize"):
        search_query = (
            await _contextualize_query(
                query,
                query_transform_model,
                history_text,
                credentials,
            )
            if transform_available
            else query
        )

    original_embedding_task = asyncio.create_task(
        _embed_variant(
            search_query,
            embedding_model,
            is_document=False,
            credentials=credentials,
        ),
    )

    try:
        with timed("retrieval.query_transform"):
            variant_bundle = await build_query_variants(
                search_query,
                query_transform_model,
                effective_transform_mode,
                credentials,
            )
    except asyncio.CancelledError:
        original_embedding_task.cancel()
        await asyncio.gather(original_embedding_task, return_exceptions=True)
        raise
    except Exception as e:
        original_embedding_task.cancel()
        await asyncio.gather(original_embedding_task, return_exceptions=True)
        raise RetrievalError("Failed during query transform / HyDE generation") from e

    logger.info(
        "Query transform mode %s produced variants: %s",
        effective_transform_mode.value,
        [(variant.kind, len(variant.text)) for variant in variant_bundle.variants],
    )

    budget = retrieval_budget(effective_transform_mode, initial_k)

    _stage("retrieve")

    try:
        with timed("retrieval.embed"):
            embeddings = await asyncio.gather(
                *(
                    _embedding_for_variant(
                        variant.kind,
                        variant.text,
                        embedding_model,
                        is_document=variant.is_document,
                        original_embedding_task=original_embedding_task,
                        credentials=credentials,
                    )
                    for variant in variant_bundle.variants
                ),
            )

        with timed("retrieval.vector_search"):
            search_results = await asyncio.gather(
                *(
                    _search_both(db, embedding, embedding_model, budget.per_query_k)
                    for embedding in embeddings
                ),
            )

        candidates = _build_candidates(
            [questions for questions, _chunks in search_results],
            [chunks for _questions, chunks in search_results],
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
            return RetrievedContext(text="")

    except Exception as e:
        raise RetrievalError("Failed during multi-query vector search") from e

    rerank_texts = [c.rerank_text for c in candidates]

    _stage("rerank")

    try:
        logger.info("Sending %d candidates to re-ranker...", len(rerank_texts))

        with timed("retrieval.rerank"):
            response = await _post_rerank(
                {"query": variant_bundle.rerank_query, "documents": rerank_texts},
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
        scores_by_key: dict[str, float] = {}
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
            candidate = candidates[idx]
            score = item["score"]
            scores_by_key[candidate.key] = score
            if score < settings.RERANKER_MIN_SCORE:
                dropped.append(
                    (candidate.key, candidate.source, score),
                )
                continue
            ranked_candidates.append(candidate)

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
        sources = visible_sources(
            [
                (candidate.retrieval_source, scores_by_key.get(candidate.key))
                for candidate in final
            ],
            source_score_floor=settings.SOURCE_RERANKER_MIN_SCORE,
        )

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
        sources = ()

    record_retrieval_ids([c.key for c in final[:_RETRIEVAL_IDS_CAP]])

    _stage("context")

    with timed("retrieval.expand"):
        text = await _expand_and_render(db, final)
    return RetrievedContext(
        text=text,
        sources=sources,
    )


async def _contextualize_query(
    query: str,
    query_transform_model: ChatModel,
    history_text: str | None,
    credentials: LlmProviderCredentials | None = None,
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
            credentials=credentials,
        )
    except Exception:
        logger.exception("Query contextualization failed; using the raw query")
        return query
    condensed = condensed.strip()
    if condensed and condensed != query:
        logger.info(
            "Contextualized query: query_len=%d condensed_len=%d history_char_len=%d",
            len(query),
            len(condensed),
            len(history_text),
        )
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
