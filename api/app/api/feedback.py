import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.feedback import (
    retract_web_feedback,
    upsert_feedback,
    upsert_web_feedback,
)
from app.schemas.feedback import (
    FeedbackAckSchema,
    FeedbackRetractionAckSchema,
    FeedbackRetractionSchema,
    FeedbackSchema,
)
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
    feedback = payload
    ack: FeedbackAckSchema | None
    if payload.client == "web":
        stored = await upsert_web_feedback(db, payload)
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Response not found",
            )
        feedback = stored.feedback
        ack = stored.ack
    else:
        ack = await upsert_feedback(db, feedback)

    if ack is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback",
        )

    capture(
        str(feedback.user_id),
        "chat_feedback",
        {
            "response_id": str(feedback.response_id),
            "client": feedback.client,
            "feedback_type": feedback.feedback_type,
            "inference_model": feedback.inference_model,
            "embeddings_model": feedback.embeddings_model,
            "query_transform_model": feedback.query_transform_model,
        },
    )

    logger.info(
        "Recorded %s feedback from %s for response %s",
        feedback.feedback_type,
        feedback.client,
        feedback.response_id,
    )

    return ack


@router.delete(
    "/feedback",
    summary="Retract response feedback",
    description="Remove the authenticated web user's feedback for an owned response.",
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid or missing API Key",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Response not found",
        },
    },
    dependencies=[api_key_dep],
    operation_id="retractFeedback",
)
async def retract_feedback(
    payload: FeedbackRetractionSchema,
    db: Database = db_dep,
) -> FeedbackRetractionAckSchema:
    retracted = await retract_web_feedback(
        db,
        response_id=payload.response_id,
        user_id=payload.user_id,
    )
    if not retracted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Response not found",
        )

    capture(
        payload.user_id,
        "chat_feedback_retracted",
        {
            "response_id": str(payload.response_id),
            "client": payload.client,
        },
    )
    logger.info(
        "Retracted feedback from %s for response %s",
        payload.client,
        payload.response_id,
    )
    return FeedbackRetractionAckSchema(response_id=payload.response_id)
