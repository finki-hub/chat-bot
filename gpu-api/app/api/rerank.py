import asyncio
import logging

from fastapi import APIRouter, status

from app.llms.reranker import rerank_documents
from app.schemas.rerank import RankedDocument, RerankRequestSchema, RerankResponseSchema

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rerank",
    tags=["Re-ranking"],
)


@router.post(
    "/",
    summary="Re-rank documents based on a query",
    description=(
        "Accepts a query and a list of documents, and returns them re-ordered "
        "by their semantic relevance to the query."
    ),
    response_model=RerankResponseSchema,
    status_code=status.HTTP_200_OK,
    operation_id="rerankDocuments",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "An unexpected error occurred during the re-ranking process.",
            "content": {
                "application/json": {
                    "example": {"detail": "An unexpected error occurred."},
                },
            },
        },
    },
)
async def handle_rerank(payload: RerankRequestSchema) -> RerankResponseSchema:
    logger.info(
        "Received rerank request with query: %s and %d documents",
        payload.query,
        len(payload.documents),
    )

    if not payload.documents:
        return RerankResponseSchema(reranked_documents=[])

    scored_list = await asyncio.to_thread(
        rerank_documents,
        payload.query,
        payload.documents,
    )

    return RerankResponseSchema(
        reranked_documents=[
            RankedDocument(document=doc, score=score) for score, doc in scored_list
        ],
    )
