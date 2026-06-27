import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.feedback import upsert_feedback
from app.schemas.feedback import FeedbackAckSchema, FeedbackSchema
from app.utils.auth import verify_api_key
from app.utils.posthog_client import capture

logger = logging.getLogger(__name__)

db_dep = Depends(get_db)
api_key_dep = Depends(verify_api_key)

router = APIRouter(
    prefix="/chat",
    tags=["Feedback"],
    dependencies=[db_dep],
)


@router.post(
    "/feedback",
    summary="Submit response feedback",
    description=(
        "Record a like/dislike for a chat response, identified by the response_id "
        "returned in the X-Response-Id header of POST /chat. Upserts on "
        "(response_id, client, user_id): re-submitting the opposite rating flips it."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid or missing API Key",
        },
    },
    dependencies=[api_key_dep],
    operation_id="submitFeedback",
)
async def submit_feedback(
    payload: FeedbackSchema,
    db: Database = db_dep,
) -> FeedbackAckSchema:
    ack = await upsert_feedback(db, payload)

    if ack is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback",
        )

    # Metadata only: question_text / answer_text are deliberately excluded (residency).
    capture(
        str(payload.user_id),
        "chat_feedback",
        {
            "response_id": str(payload.response_id),
            "client": payload.client,
            "feedback_type": payload.feedback_type,
            "inference_model": payload.inference_model,
            "embeddings_model": payload.embeddings_model,
            "query_transform_model": payload.query_transform_model,
        },
    )

    logger.info(
        "Recorded %s feedback from %s for response %s",
        payload.feedback_type,
        payload.client,
        payload.response_id,
    )

    return ack
