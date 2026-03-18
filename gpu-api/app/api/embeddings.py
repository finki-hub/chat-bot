from fastapi import APIRouter, status

from app.llms.embeddings import generate_embeddings
from app.schemas.embeddings import EmbedRequestSchema, EmbedResponseSchema

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
    embeddings = await generate_embeddings(payload.input, payload.embeddings_model)
    return EmbedResponseSchema(embeddings=embeddings)
