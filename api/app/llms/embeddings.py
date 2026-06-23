import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import overload

from asyncpg import Record
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.data.documents import fetch_chunk_rows_for_fill
from app.llms.google import generate_google_embeddings
from app.llms.gpu_api import generate_gpu_api_embeddings
from app.llms.models import (
    ALL_MODELS_EMBEDDINGS,
    MODEL_EMBEDDING_DIMENSIONS,
    MODEL_EMBEDDINGS_COLUMNS,
    Model,
)
from app.llms.ollama import generate_ollama_embeddings
from app.llms.openai import generate_openai_embeddings
from app.llms.text_utils import _prepare_text_for_embedding
from app.utils.database import embedding_to_pgvector

logger = logging.getLogger(__name__)


EMBEDDING_MAX_ATTEMPTS = 3
_EMBEDDING_RETRY_BASE_DELAY = 0.5


async def _dispatch_embeddings(
    text: str | list[str],
    model: Model,
    *,
    is_document: bool,
) -> list[float] | list[list[float]]:
    match model:
        case Model.LLAMA_3_3_70B | Model.BGE_M3:
            return await generate_ollama_embeddings(text, model)

        case Model.TEXT_EMBEDDING_3_LARGE:
            return await generate_openai_embeddings(text, model)

        case Model.GEMINI_EMBEDDING_001:
            return await generate_google_embeddings(
                text,
                model,
                is_document=is_document,
            )

        case Model.MULTILINGUAL_E5_LARGE | Model.BGE_M3_LOCAL:
            return await generate_gpu_api_embeddings(text, model)

        case _:
            raise ValueError(f"Unsupported model: {model}")


@overload
async def generate_embeddings(
    text: str,
    model: Model,
    *,
    is_document: bool = ...,
) -> list[float]: ...


@overload
async def generate_embeddings(
    text: list[str],
    model: Model,
    *,
    is_document: bool = ...,
) -> list[list[float]]: ...


async def generate_embeddings(
    text: str | list[str],
    model: Model,
    *,
    is_document: bool = False,
) -> list[float] | list[list[float]]:
    """
    Generate embeddings for the given text using the specified model.
    Pass is_document=True when indexing documents (only affects Gemini task_type).

    Transient provider failures (a 429 / 5xx / timeout during a full-corpus fill) are
    retried with bounded exponential backoff so a single blip does not fail a whole
    batch. An unsupported model is a permanent ValueError and is not retried.
    """

    log_preview = (
        text[:100] if isinstance(text, str) else f"[list of {len(text)} items]"
    )
    logger.info("Generating embeddings for text: '%s'", log_preview)

    for attempt in range(1, EMBEDDING_MAX_ATTEMPTS + 1):
        try:
            return await _dispatch_embeddings(text, model, is_document=is_document)
        except ValueError:
            raise
        except Exception:
            if attempt >= EMBEDDING_MAX_ATTEMPTS:
                logger.exception(
                    "Embedding generation failed for model %s after %d attempts",
                    model.value,
                    attempt,
                )
                raise

            delay = _EMBEDDING_RETRY_BASE_DELAY * 2 ** (attempt - 1)
            logger.warning(
                "Embedding attempt %d/%d for model %s failed; retrying in %.1fs",
                attempt,
                EMBEDDING_MAX_ATTEMPTS,
                model.value,
                delay,
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        "Unreachable: embedding retry loop exited without return or raise",
    )


def _resolve_models(model: Model, *, all_models: bool) -> list[Model]:
    if all_models:
        return list(ALL_MODELS_EMBEDDINGS)

    if model not in MODEL_EMBEDDINGS_COLUMNS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported embedding model: {model.value}",
        )
    return [model]


async def _fetch_question_rows(
    db: Database,
    questions: list[str] | None,
    *,
    all_questions: bool,
) -> list[Record]:
    if all_questions:
        return await db.fetch("SELECT id, name, content FROM question")

    if questions:
        placeholders = ",".join(["$" + str(i + 1) for i in range(len(questions))])
        return await db.fetch(
            f"SELECT id, name, content FROM question WHERE name IN ({placeholders})",  # noqa: S608
            *questions,
        )

    return []


async def _count_unfilled_tasks(
    db: Database,
    models: list[Model],
) -> int:
    total = 0
    for m in models:
        col = MODEL_EMBEDDINGS_COLUMNS[m]
        count_result = await db.fetchval(
            f"SELECT COUNT(*) FROM question WHERE {col} IS NULL",  # noqa: S608
        )
        if isinstance(count_result, int | str):
            total += int(count_result)
    return total


EMBEDDING_BATCH_SIZE = 16


async def _process_question_batch(
    db: Database,
    batch: list[Record],
    current_model: Model,
    model_column: str,
) -> list[tuple[str, str]]:
    texts = []
    for row in batch:
        document_text = f"Наслов: {row['name']}\nСодржина: {row['content']}"
        texts.append(
            _prepare_text_for_embedding(document_text, current_model, is_document=True),
        )

    try:
        embeddings = await generate_embeddings(
            texts,
            current_model,
            is_document=True,
        )
    except Exception as e:
        return [("error", repr(e))] * len(batch)

    results: list[tuple[str, str]] = []
    expected_dims = MODEL_EMBEDDING_DIMENSIONS.get(current_model)
    for row, embedding in zip(batch, embeddings, strict=True):
        if expected_dims is not None and len(embedding) != expected_dims:
            results.append(
                (
                    "error",
                    f"Dimension mismatch: got {len(embedding)}, expected {expected_dims}",
                ),
            )
            continue
        try:
            await db.execute(
                f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
                embedding_to_pgvector(embedding),
                row["id"],
            )
            results.append(("ok", ""))
        except Exception as e:
            results.append(("error", repr(e)))

    return results


async def stream_fill_embeddings(
    db: Database,
    model: Model,
    *,
    questions: list[str] | None = None,
    all_questions: bool = False,
    all_models: bool = False,
) -> StreamingResponse:
    """
    Stream progress of filling embeddings for questions.
    Can process a single model or all available embedding models.
    Emits one SSE event per question-model combination as JSON.
    """

    logger.info(
        "Starting to fill embeddings for model: %s, all_questions: %s, all_models: %s",
        model,
        all_questions,
        all_models,
    )

    models_to_process = _resolve_models(model, all_models=all_models)
    question_rows = await _fetch_question_rows(
        db,
        questions,
        all_questions=all_questions,
    )

    if question_rows:
        total_tasks = len(question_rows) * len(models_to_process)
    else:
        total_tasks = await _count_unfilled_tasks(db, models_to_process)

    async def _gen() -> AsyncGenerator[str]:
        progress_counter = 0

        for current_model in models_to_process:
            model_column = MODEL_EMBEDDINGS_COLUMNS[current_model]

            rows_for_this_model = question_rows or await db.fetch(
                f"SELECT id, name, content FROM question WHERE {model_column} IS NULL",  # noqa: S608
            )

            for batch_start in range(0, len(rows_for_this_model), EMBEDDING_BATCH_SIZE):
                batch = rows_for_this_model[
                    batch_start : batch_start + EMBEDDING_BATCH_SIZE
                ]
                results = await _process_question_batch(
                    db,
                    batch,
                    current_model,
                    model_column,
                )

                for row, (result, error_detail) in zip(batch, results, strict=True):
                    progress_counter += 1
                    payload = {
                        "status": result,
                        "error": error_detail,
                        "index": progress_counter,
                        "total": total_tasks,
                        "model": current_model.value,
                        "id": str(row["id"]),
                        "name": row["name"],
                        "ts": datetime.now(UTC).isoformat() + "Z",
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
    )


def _chunk_document_text(row: Record) -> str:
    title = row["document_title"]
    section = row["section"]
    label = f"{title} ({section})" if section else title
    return f"Наслов: {label}\nСодржина: {row['content']}"


async def _process_chunk_batch(
    db: Database,
    batch: list[Record],
    current_model: Model,
    model_column: str,
) -> list[tuple[str, str]]:
    texts = [
        _prepare_text_for_embedding(
            _chunk_document_text(row),
            current_model,
            is_document=True,
        )
        for row in batch
    ]

    try:
        embeddings = await generate_embeddings(texts, current_model, is_document=True)
    except Exception as e:
        return [("error", repr(e))] * len(batch)

    results: list[tuple[str, str]] = []
    expected_dims = MODEL_EMBEDDING_DIMENSIONS.get(current_model)
    for row, embedding in zip(batch, embeddings, strict=True):
        if expected_dims is not None and len(embedding) != expected_dims:
            results.append(
                (
                    "error",
                    f"Dimension mismatch: got {len(embedding)}, expected {expected_dims}",
                ),
            )
            continue
        try:
            await db.execute(
                f"UPDATE chunk SET {model_column} = $1 WHERE id = $2",  # noqa: S608
                embedding_to_pgvector(embedding),
                row["id"],
            )
            results.append(("ok", ""))
        except Exception as e:
            results.append(("error", repr(e)))

    return results


async def stream_fill_chunk_embeddings(
    db: Database,
    model: Model,
    *,
    documents: list[str] | None = None,
    all_chunks: bool = False,
    all_models: bool = False,
) -> StreamingResponse:
    """Stream per-chunk embedding-fill progress as SSE (analogue of stream_fill_embeddings)."""

    logger.info(
        "Filling chunk embeddings for model: %s, all_chunks: %s, all_models: %s",
        model,
        all_chunks,
        all_models,
    )

    models_to_process = _resolve_models(model, all_models=all_models)

    async def _gen() -> AsyncGenerator[str]:
        per_model_rows: dict[Model, list[Record]] = {}
        total_tasks = 0
        for current_model in models_to_process:
            model_column = MODEL_EMBEDDINGS_COLUMNS[current_model]
            rows = await fetch_chunk_rows_for_fill(
                db,
                model_column,
                documents,
                all_chunks=all_chunks,
            )
            per_model_rows[current_model] = rows
            total_tasks += len(rows)

        progress_counter = 0
        for current_model in models_to_process:
            model_column = MODEL_EMBEDDINGS_COLUMNS[current_model]
            rows = per_model_rows[current_model]

            for batch_start in range(0, len(rows), EMBEDDING_BATCH_SIZE):
                batch = rows[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                results = await _process_chunk_batch(
                    db,
                    batch,
                    current_model,
                    model_column,
                )

                for row, (result, error_detail) in zip(batch, results, strict=True):
                    progress_counter += 1
                    payload = {
                        "status": result,
                        "error": error_detail,
                        "index": progress_counter,
                        "total": total_tasks,
                        "model": current_model.value,
                        "id": str(row["id"]),
                        "name": row["document_title"],
                        "ts": datetime.now(UTC).isoformat() + "Z",
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
    )
