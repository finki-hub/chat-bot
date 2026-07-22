import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime

from asyncpg import PostgresError, Record
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.diplomas import fetch_diploma_rows_for_fill
from app.data.documents import fetch_chunk_rows_for_fill
from app.data.embedding_lifecycle import (
    EmbeddingBatch,
    EmbeddingCandidate,
    embedding_candidate_from_row,
    persist_embedding_batch,
)
from app.data.embedding_lifecycle_sql import EmbeddingCorpus
from app.data.professor_documents import fetch_professor_document_rows_for_fill
from app.data.questions import fetch_question_rows_for_fill
from app.llms.embedding_generation import generate_embeddings, resolve_fill_models
from app.llms.models import Model
from app.llms.text_utils import _prepare_text_for_embedding

logger = logging.getLogger(__name__)
_EMBEDDING_SSE_ERROR = "Embedding processing failed."
EMBEDDING_BATCH_SIZE = 16

type RowFetcher = Callable[[Database, Model], Awaitable[list[Record]]]
type RowName = Callable[[Record], str]


async def _process_batch(
    database: Database,
    candidates: tuple[EmbeddingCandidate, ...],
    model: Model,
) -> list[tuple[str, str]]:
    texts = [
        _prepare_text_for_embedding(candidate.text, model, is_document=True)
        for candidate in candidates
    ]
    try:
        vectors = await generate_embeddings(texts, model, is_document=True)
    except Exception:
        return [("error", _EMBEDDING_SSE_ERROR)] * len(candidates)

    try:
        result = await persist_embedding_batch(
            database,
            EmbeddingBatch(
                corpus=candidates[0].corpus,
                candidates=candidates,
                vectors=tuple(tuple(vector) for vector in vectors),
            ),
        )
    except PostgresError:
        return [("error", _EMBEDDING_SSE_ERROR)] * len(candidates)
    if not result.valid:
        return [("error", _EMBEDDING_SSE_ERROR)] * len(candidates)
    return [
        ("ok", "") if applied else ("error", _EMBEDDING_SSE_ERROR)
        for applied in result.applied
    ]


def _stream_fill(
    database: Database,
    model: Model,
    *,
    corpus: EmbeddingCorpus,
    fetch_rows: RowFetcher,
    row_name: RowName,
    all_models: bool,
) -> StreamingResponse:
    logger.info("Filling %s embeddings for model: %s", corpus.value, model)
    models_to_process = resolve_fill_models(model, all_models=all_models)

    async def generate() -> AsyncGenerator[str]:
        candidates_by_model: dict[Model, tuple[EmbeddingCandidate, ...]] = {}
        names_by_model: dict[Model, tuple[str, ...]] = {}
        total_tasks = 0
        for current_model in models_to_process:
            rows = await fetch_rows(database, current_model)
            candidates = tuple(
                embedding_candidate_from_row(corpus, row) for row in rows
            )
            candidates_by_model[current_model] = candidates
            names_by_model[current_model] = tuple(row_name(row) for row in rows)
            total_tasks += len(candidates)

        progress = 0
        for current_model in models_to_process:
            candidates = candidates_by_model[current_model]
            for batch_start in range(0, len(candidates), EMBEDDING_BATCH_SIZE):
                batch = candidates[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                names = names_by_model[current_model][
                    batch_start : batch_start + EMBEDDING_BATCH_SIZE
                ]
                results = await _process_batch(database, batch, current_model)
                for candidate, (status, error), name in zip(
                    batch,
                    results,
                    names,
                    strict=True,
                ):
                    progress += 1
                    payload = {
                        "status": status,
                        "error": error,
                        "index": progress,
                        "total": total_tasks,
                        "model": current_model.value,
                        "id": str(candidate.id),
                        "name": name,
                        "ts": datetime.now(UTC).isoformat() + "Z",
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


async def stream_fill_embeddings(
    database: Database,
    model: Model,
    *,
    questions: list[str] | None = None,
    all_questions: bool = False,
    all_models: bool = False,
) -> StreamingResponse:
    async def fetch_rows(db: Database, current_model: Model) -> list[Record]:
        return await fetch_question_rows_for_fill(
            db,
            current_model,
            questions,
            all_questions=all_questions,
        )

    return _stream_fill(
        database,
        model,
        corpus=EmbeddingCorpus.QUESTION,
        fetch_rows=fetch_rows,
        row_name=lambda row: str(row["name"]),
        all_models=all_models,
    )


async def stream_fill_chunk_embeddings(
    database: Database,
    model: Model,
    *,
    documents: list[str] | None = None,
    all_chunks: bool = False,
    all_models: bool = False,
) -> StreamingResponse:
    async def fetch_rows(db: Database, current_model: Model) -> list[Record]:
        return await fetch_chunk_rows_for_fill(
            db,
            current_model,
            documents,
            all_chunks=all_chunks,
        )

    return _stream_fill(
        database,
        model,
        corpus=EmbeddingCorpus.CHUNK,
        fetch_rows=fetch_rows,
        row_name=lambda row: str(row["document_title"]),
        all_models=all_models,
    )


async def stream_fill_professor_document_embeddings(
    database: Database,
    model: Model,
    *,
    all_models: bool = False,
) -> StreamingResponse:
    return _stream_fill(
        database,
        model,
        corpus=EmbeddingCorpus.PROFESSOR_DOCUMENT,
        fetch_rows=fetch_professor_document_rows_for_fill,
        row_name=lambda row: str(row["title"]),
        all_models=all_models,
    )


async def stream_fill_diploma_embeddings(
    database: Database,
    model: Model,
    *,
    all_models: bool = False,
) -> StreamingResponse:
    return _stream_fill(
        database,
        model,
        corpus=EmbeddingCorpus.DIPLOMA,
        fetch_rows=fetch_diploma_rows_for_fill,
        row_name=lambda row: str(row["title"]),
        all_models=all_models,
    )
