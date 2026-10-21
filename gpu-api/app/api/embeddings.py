import json
import logging

from fastapi import APIRouter, Request, status

from app.llms.embeddings import generate_embeddings
from app.schemas.embeddings import EmbedRequestSchema, EmbedResponseSchema
from app.utils.analytics import capture_chat_inference

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/embeddings",
    tags=["Embeddings"],
)


@router.post(
    "/embed",
    summary="Generate embeddings",
    description="Given text(s) and a model, return the embedding vector(s).",
    status_code=status.HTTP_200_OK,
    response_description="The embedding(s) as a list of floats",
    operation_id="embedText",
)
async def embed(payload: EmbedRequestSchema, request: Request) -> EmbedResponseSchema:
    run = await generate_embeddings(payload.input, payload.embeddings_model)
    count = 1 if isinstance(payload.input, str) else len(payload.input)
    input_chars = (
        len(payload.input)
        if isinstance(payload.input, str)
        else sum(len(item) for item in payload.input)
    )
    logger.info(
        "gpu.embed %s",
        json.dumps(
            {
                "model": payload.embeddings_model.value,
                "count": count,
                "ms": run.run_ms,
                "queue_ms": run.queue_ms,
            },
        ),
    )
    capture_chat_inference(
        request,
        stage="embed",
        ms=run.run_ms,
        props={
            "model": payload.embeddings_model.value,
            "count": count,
            "input_chars": input_chars,
            "queue_ms": run.queue_ms,
        },
    )
    return EmbedResponseSchema(embeddings=run.value)
