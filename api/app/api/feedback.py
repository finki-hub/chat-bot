import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app.data.connection import Database
from app.data.db import get_db
from app.data.feedback import (
    DiscordFeedbackStaleError,
    DiscordFeedbackUnavailableError,
    retract_web_feedback,
    upsert_discord_feedback,
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
        " Discord requires a registered rater and a surviving stored response; "
        "the rater need not own it. Copied question/answer/model values are ignored."
        " Retry 404 only when X-Feedback-Generation is returned, echoing that same "
        "immutable decimal header on every retry. Never refresh it. 409 is terminal. "
        "This internal deletion-fence token is not an authentication token."
    ),
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Invalid or missing API Key",
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "Response unavailable. A registered, unchanged Discord rater receives an immutable retry generation; missing accounts receive no token.",
            "headers": {
                "X-Feedback-Generation": {
                    "description": "Echo unchanged on bounded retries; never refresh across attempts.",
                    "schema": {
                        "type": "string",
                        "pattern": "^(0|[1-9][0-9]*)$",
                        "maxLength": 19,
                    },
                },
            },
        },
        status.HTTP_409_CONFLICT: {
            "description": "Rater or captured response-owner history changed since the first attempt/token. Do not retry or refresh the token.",
        },
    },
    dependencies=[api_key_dep],
    operation_id="submitFeedback",
)
async def submit_feedback(
    payload: FeedbackSchema,
    db: Database = db_dep,
    x_feedback_generation: Annotated[
        str | None,
        Header(
            alias="X-Feedback-Generation",
            min_length=1,
            max_length=19,
            pattern=r"^(0|[1-9][0-9]*)$",
        ),
    ] = None,
) -> FeedbackAckSchema:
    feedback = payload
    ack: FeedbackAckSchema | None
    try:
        stored = (
            await upsert_web_feedback(db, payload)
            if payload.client == "web"
            else await upsert_discord_feedback(
                db,
                payload,
                generation=int(x_feedback_generation)
                if x_feedback_generation is not None
                else None,
            )
        )
    except DiscordFeedbackStaleError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Feedback history changed; do not retry",
        ) from None
    except DiscordFeedbackUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Response or rater unavailable",
            headers={"X-Feedback-Generation": str(exc.generation)}
            if exc.generation is not None
            else None,
        ) from None
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Response not found or no longer available",
        )
    feedback = stored.feedback
    ack = stored.ack

    capture(
        "feedback",
        "chat_feedback",
        {
            "response_id": str(feedback.response_id),
            "$process_person_profile": False,
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
        "feedback",
        "chat_feedback_retracted",
        {
            "response_id": str(payload.response_id),
            "$process_person_profile": False,
            "client": payload.client,
        },
    )
    logger.info(
        "Retracted feedback from %s for response %s",
        payload.client,
        payload.response_id,
    )
    return FeedbackRetractionAckSchema(response_id=payload.response_id)
