import asyncio
import json
import logging
from time import perf_counter

import torch
from fastapi import APIRouter, Request, status

from app.llms.reranker import rerank_documents
from app.schemas.rerank import RankedDocument, RerankRequestSchema, RerankResponseSchema
from app.utils.analytics import capture, safe_response_id

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
async def handle_rerank(
    payload: RerankRequestSchema,
    request: Request,
) -> RerankResponseSchema:
    logger.info(
        "Received rerank request with query: %s and %d documents",
        payload.query,
        len(payload.documents),
    )

    if not payload.documents:
        return RerankResponseSchema(reranked_documents=[])

    start = perf_counter()
    scored_list = await asyncio.to_thread(
        rerank_documents,
        payload.query,
        payload.documents,
    )
    ms = round((perf_counter() - start) * 1000, 1)
    logger.info(
        "gpu.rerank %s",
        json.dumps(
            {
                "docs": len(payload.documents),
                "ms": ms,
            },
        ),
    )
    response_id = safe_response_id(request.headers.get("X-Response-Id"))
    capture(
        response_id or "gpu-api",
        "chat_inference",
        {
            "stage": "rerank",
            "docs": len(payload.documents),
            "query_chars": len(payload.query),
            "doc_chars": sum(len(doc) for doc in payload.documents),
            "ms": ms,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "response_id": response_id,
        },
    )

    return RerankResponseSchema(
        reranked_documents=[
            RankedDocument(index=idx, document=payload.documents[idx], score=score)
            for score, idx in scored_list
        ],
    )
