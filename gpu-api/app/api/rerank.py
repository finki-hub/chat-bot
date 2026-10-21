import json
import logging

from fastapi import APIRouter, Request, status

from app.llms.inference import Lane, run_in_lane
from app.llms.reranker import rerank_documents
from app.schemas.rerank import RankedDocument, RerankRequestSchema, RerankResponseSchema
from app.utils.analytics import capture_chat_inference

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
        "Received rerank request: query_len=%d docs=%d",
        len(payload.query),
        len(payload.documents),
    )

    if not payload.documents:
        return RerankResponseSchema(reranked_documents=[])

    run = await run_in_lane(
        Lane.RERANK,
        rerank_documents,
        payload.query,
        payload.documents,
    )
    logger.info(
        "gpu.rerank %s",
        json.dumps(
            {
                "docs": len(payload.documents),
                "ms": run.run_ms,
                "queue_ms": run.queue_ms,
            },
        ),
    )
    capture_chat_inference(
        request,
        stage="rerank",
        ms=run.run_ms,
        props={
            "docs": len(payload.documents),
            "query_chars": len(payload.query),
            "doc_chars": sum(len(doc) for doc in payload.documents),
            "queue_ms": run.queue_ms,
        },
    )

    return RerankResponseSchema(
        reranked_documents=[
            RankedDocument(index=idx, document=payload.documents[idx], score=score)
            for score, idx in run.value
        ],
    )
