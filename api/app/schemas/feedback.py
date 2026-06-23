from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class FeedbackSchema(BaseModel):
    response_id: UUID = Field(
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="Canonical id of the rated response, as returned in the X-Response-Id header of POST /chat.",
    )
    client: Literal["discord", "web"] = Field(
        examples=["discord"],
        description="Which consumer is submitting the feedback.",
    )
    user_id: str = Field(
        min_length=1,
        examples=["198249751001563136"],
        description="Id of the user giving feedback, namespaced per client (e.g. Discord user id).",
    )
    feedback_type: Literal["like", "dislike"] = Field(
        examples=["like"],
        description="The rating.",
    )
    client_ref: str | None = Field(
        default=None,
        examples=["1380000000000000000"],
        description="Optional client-side handle for traceability (e.g. the Discord message id).",
    )
    channel_id: str | None = Field(
        default=None,
        description="Optional channel id (Discord).",
    )
    guild_id: str | None = Field(
        default=None,
        description="Optional guild id (Discord).",
    )
    question_text: str | None = Field(
        default=None,
        description="The user's question, as reported by the consumer (client-attested, not server-verified).",
    )
    answer_text: str | None = Field(
        default=None,
        description="The rated answer, as reported by the consumer (client-attested, not server-verified).",
    )
    inference_model: str | None = Field(
        default=None,
        description="Inference model used to generate the answer, if known.",
    )
    embeddings_model: str | None = Field(
        default=None,
        description="Embeddings model used for retrieval, if known.",
    )
    query_transform_model: str | None = Field(
        default=None,
        description="Query-transform model used, if known.",
    )


class FeedbackAckSchema(BaseModel):
    id: UUID = Field(
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="Unique identifier of the stored feedback row.",
    )
    response_id: UUID = Field(
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
        description="The response the feedback applies to.",
    )
    feedback_type: Literal["like", "dislike"] = Field(
        examples=["like"],
        description="The recorded rating after the upsert.",
    )
