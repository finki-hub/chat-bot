import json
import logging
from time import perf_counter

import torch
from fastapi import APIRouter, Request, status

from app.llms.embeddings import generate_embeddings
from app.schemas.embeddings import EmbedRequestSchema, EmbedResponseSchema
from app.utils.analytics import capture, safe_response_id

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
    responses={
        status.HTTP_404_NOT_FOUND: {
            "description": "Unsupported model or invalid input",
        },
    },
)
async def embed(payload: EmbedRequestSchema, request: Request) -> EmbedResponseSchema:
    start = perf_counter()
    embeddings = await generate_embeddings(payload.input, payload.embeddings_model)
    count = 1 if isinstance(payload.input, str) else len(payload.input)
    input_chars = (
        len(payload.input)
        if isinstance(payload.input, str)
        else sum(len(item) for item in payload.input)
    )
    ms = round((perf_counter() - start) * 1000, 1)
    logger.info(
        "gpu.embed %s",
        json.dumps(
            {
                "model": payload.embeddings_model.value,
                "count": count,
                "ms": ms,
            },
        ),
    )
    response_id = safe_response_id(request.headers.get("X-Response-Id"))
    capture(
        response_id or "gpu-api",
        "chat_inference",
        {
            "stage": "embed",
            "model": payload.embeddings_model.value,
            "count": count,
            "input_chars": input_chars,
            "ms": ms,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "response_id": response_id,
        },
    )
    return EmbedResponseSchema(embeddings=embeddings)
