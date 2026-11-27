from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.defaults import DEFAULT_QUERY_TRANSFORM_MODEL
from app.llms.models import QUERY_TRANSFORM_MODELS, ChatModel, parse_chat_model

MAX_SUGGESTIONS = 3
MAX_SUGGESTION_LENGTH = 100


class ChatSuggestionsSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID | None = Field(
        None,
        description="Authenticated chat user id used to resolve per-user provider credentials.",
    )
    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question the answer replied to.",
    )
    answer: str = Field(
        min_length=1,
        max_length=8000,
        description="The finished answer to suggest follow-up questions for.",
    )
    query_transform_model: ChatModel | None = Field(
        None,
        description="Optional chat-capable model override for writing the suggestions.",
    )
    provider_model: ChatModel = Field(
        DEFAULT_QUERY_TRANSFORM_MODEL,
        description="Active model whose provider should be used for automatic selection.",
    )

    @field_validator("provider_model")
    @classmethod
    def _provider_must_be_chat_capable(cls, value: ChatModel) -> ChatModel:
        try:
            return parse_chat_model(value, QUERY_TRANSFORM_MODELS)
        except ValueError as error:
            raise ValueError("provider_model must be a chat-capable model") from error

    @field_validator("query_transform_model")
    @classmethod
    def _override_must_be_chat_capable(
        cls,
        value: ChatModel | None,
    ) -> ChatModel | None:
        if value is None:
            return None
        try:
            return parse_chat_model(value, QUERY_TRANSFORM_MODELS)
        except ValueError as error:
            raise ValueError(
                "query_transform_model must be a chat-capable model",
            ) from error


class ChatSuggestionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    suggestions: list[str] = Field(
        max_length=MAX_SUGGESTIONS,
        description="Follow-up questions the knowledge base can answer; may be empty.",
    )
