import json
import logging
from time import perf_counter

from fastapi import APIRouter, status

from app.llms.embeddings import generate_embeddings
from app.schemas.embeddings import EmbedRequestSchema, EmbedResponseSchema

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
async def embed(payload: EmbedRequestSchema) -> EmbedResponseSchema:
    start = perf_counter()
    embeddings = await generate_embeddings(payload.input, payload.embeddings_model)
    count = 1 if isinstance(payload.input, str) else len(payload.input)
    logger.info(
        "gpu.embed %s",
        json.dumps(
            {
                "model": payload.embeddings_model.value,
                "count": count,
                "ms": round((perf_counter() - start) * 1000, 1),
            },
        ),
    )
    return EmbedResponseSchema(embeddings=embeddings)
