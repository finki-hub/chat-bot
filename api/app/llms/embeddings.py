import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import overload

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms.google import generate_google_embeddings
from app.llms.gpu_api import generate_gpu_api_embeddings
from app.llms.models import MODEL_EMBEDDINGS_COLUMNS, Model
from app.llms.ollama import generate_ollama_embeddings
from app.llms.openai import generate_openai_embeddings
from app.utils.database import embedding_to_pgvector

logger = logging.getLogger(__name__)


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
    """

    log_preview = text[:100] if isinstance(text, str) else f"[list of {len(text)} items]"
    logger.info("Generating embeddings for text: '%s'", log_preview)

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

    models_to_process: list[Model]
    if all_models:
        models_to_process = list(MODEL_EMBEDDINGS_COLUMNS.keys())
    else:
        if model not in MODEL_EMBEDDINGS_COLUMNS:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported embedding model: {model.value}",
            )
        models_to_process = [model]

    question_rows = []
    if all_questions:
        question_rows = await db.fetch("SELECT id, name, content FROM question")
    elif questions:
        placeholders = ",".join(["$" + str(i + 1) for i in range(len(questions))])
        question_rows = await db.fetch(
            f"SELECT id, name, content FROM question WHERE name IN ({placeholders})",  # noqa: S608
            *questions,
        )

    total_tasks: int

    if question_rows:
        total_tasks = len(question_rows) * len(models_to_process)
    else:
        total_tasks = 0
        for m in models_to_process:
            col = MODEL_EMBEDDINGS_COLUMNS[m]
            count_result = await db.fetchval(
                f"SELECT COUNT(*) FROM question WHERE {col} IS NULL",  # noqa: S608
            )
            if isinstance(count_result, int | str):
                total_tasks += int(count_result)

    async def _gen() -> AsyncGenerator[str]:
        progress_counter = 0

        for current_model in models_to_process:
            model_column = MODEL_EMBEDDINGS_COLUMNS[current_model]

            rows_for_this_model = []
            if question_rows:
                rows_for_this_model = question_rows
            else:
                rows_for_this_model = await db.fetch(
                    f"SELECT id, name, content FROM question WHERE {model_column} IS NULL",  # noqa: S608
                )

            for row in rows_for_this_model:
                progress_counter += 1
                qid = row["id"]
                name = row["name"]
                result = "ok"
                error_detail = ""

                try:
                    document_text = f"Наслов: {name}\nСодржина: {row['content']}"
                    text_to_embed = (
                        f"passage: {document_text}"
                        if current_model == Model.MULTILINGUAL_E5_LARGE
                        else document_text
                    )

                    embedding = await generate_embeddings(
                        text_to_embed,
                        current_model,
                        is_document=True,
                    )
                    await db.execute(
                        f"UPDATE question SET {model_column} = $1 WHERE id = $2",  # noqa: S608
                        embedding_to_pgvector(embedding),
                        qid,
                    )
                except Exception as e:
                    result = "error"
                    error_detail = repr(e)

                payload = {
                    "status": result,
                    "error": error_detail,
                    "index": progress_counter,
                    "total": total_tasks,
                    "model": current_model.value,
                    "id": str(qid),
                    "name": name,
                    "ts": datetime.now(UTC).isoformat() + "Z",
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
    )
