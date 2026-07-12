from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants.defaults import DEFAULT_QUERY_TRANSFORM_MODEL
from app.llms.models import QUERY_TRANSFORM_MODELS, ChatModel, parse_chat_model
from app.schemas.chat import ConversationTurn


class ChatTitleSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: UUID | None = Field(
        None,
        description="Authenticated chat user id used to resolve per-user provider credentials.",
    )
    messages: list[ConversationTurn] = Field(
        min_length=1,
        max_length=4,
        description="The first conversation turns used to generate a concise title.",
    )
    query_transform_model: ChatModel = Field(
        DEFAULT_QUERY_TRANSFORM_MODEL,
        description="Which chat-capable model to use for title generation.",
    )

    @field_validator("messages")
    @classmethod
    def _must_include_user_turn(
        cls,
        value: list[ConversationTurn],
    ) -> list[ConversationTurn]:
        if not any(turn.role == "user" for turn in value):
            raise ValueError("messages must include at least one user turn")
        return value

    @field_validator("query_transform_model")
    @classmethod
    def _must_support_title_generation(cls, value: ChatModel) -> ChatModel:
        try:
            return parse_chat_model(value, QUERY_TRANSFORM_MODELS)
        except ValueError as error:
            raise ValueError(
                "query_transform_model must be a chat-capable model",
            ) from error

    @property
    def first_user_text(self) -> str:
        for turn in self.messages:
            if turn.role == "user":
                return turn.content
        return ""

    @property
    def transcript(self) -> str:
        return "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content}"
            for turn in self.messages
        )


class ChatTitleResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1, max_length=60)
