# mypy: disable-error-code="arg-type"

from app.data.connection import Database
from app.schemas.feedback import FeedbackAckSchema, FeedbackSchema


async def upsert_feedback(
    db: Database,
    feedback: FeedbackSchema,
) -> FeedbackAckSchema | None:
    query = """
    INSERT INTO feedback (
        response_id, client, user_id, feedback_type, client_ref,
        channel_id, guild_id, question_text, answer_text,
        inference_model, embeddings_model, query_transform_model
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ON CONFLICT (response_id, client, user_id)
    DO UPDATE SET feedback_type = EXCLUDED.feedback_type, updated_at = NOW()
    RETURNING id, response_id, feedback_type
    """
    result = await db.fetchrow(
        query,
        feedback.response_id,
        feedback.client,
        feedback.user_id,
        feedback.feedback_type,
        feedback.client_ref,
        feedback.channel_id,
        feedback.guild_id,
        feedback.question_text,
        feedback.answer_text,
        feedback.inference_model,
        feedback.embeddings_model,
        feedback.query_transform_model,
    )

    if not result:
        return None

    return FeedbackAckSchema(
        id=result["id"],
        response_id=result["response_id"],
        feedback_type=result["feedback_type"],
    )
