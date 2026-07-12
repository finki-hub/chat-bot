from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.constants.defaults import (
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_INFERENCE_MODEL,
    DEFAULT_QUERY_TRANSFORM_MODEL,
)
from app.llms.models import (
    ACTIVE_EMBEDDING_MODELS,
    CHAT_MODELS,
    QUERY_TRANSFORM_MODELS,
    ChatModel,
    Model,
    parse_chat_model,
)
from app.llms.query_modes import QueryTransformMode

ChatInterface = Literal["discord", "web"]


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"] = Field(
        examples=["user"],
        description="Who produced this turn: 'user' or 'assistant' (the bot).",
    )
    content: str = Field(
        max_length=2000,
        examples=["Where is FINKI located?"],
        description="The text content of this conversation turn.",
    )


class ChatSchema(BaseModel):
    user_id: UUID | None = Field(
        None,
        description="Authenticated chat user id used to resolve per-user provider credentials.",
    )
    messages: list[ConversationTurn] = Field(
        min_length=1,
        max_length=10,
        examples=[
            [
                {"role": "user", "content": "Where is FINKI located?"},
                {"role": "assistant", "content": "FINKI is located in Skopje."},
                {"role": "user", "content": "How do I get there?"},
            ],
        ],
        description=(
            "The conversation, oldest first and ending with the current user turn. "
            "The last (user) message drives retrieval; the earlier turns are passed "
            "to the model as prior context."
        ),
    )
    interface: ChatInterface = Field(
        "discord",
        examples=["discord", "web"],
        description=(
            "The client interface that will render the answer. Discord supports "
            "limited Markdown; web supports full Markdown, including tables."
        ),
    )
    embeddings_model: Model = Field(
        DEFAULT_EMBEDDINGS_MODEL,
        examples=[DEFAULT_EMBEDDINGS_MODEL.value],
        description=(
            "Which Model to use for computing embeddings for retrieval. "
            "Must be one of the values in `app.llms.models.Model`."
        ),
    )
    inference_model: ChatModel = Field(
        DEFAULT_INFERENCE_MODEL,
        examples=[DEFAULT_INFERENCE_MODEL.value],
        description=(
            "Which Model to use for generating / streaming the response. "
            "Must be one of the values in `app.llms.models.Model`."
        ),
    )
    query_transform_model: ChatModel = Field(
        DEFAULT_QUERY_TRANSFORM_MODEL,
        examples=[DEFAULT_QUERY_TRANSFORM_MODEL.value],
        description=(
            "Which Model to use for query rewriting and HyDE passage generation "
            "during retrieval. Must be a chat-capable model."
        ),
    )
    query_transform_mode: QueryTransformMode = Field(
        QueryTransformMode.REWRITE_HYDE,
        examples=[QueryTransformMode.REWRITE_HYDE.value],
        description=(
            "Which query transformation variants to add before retrieval. "
            "The raw query is always searched; rewrite and HyDE add extra variants."
        ),
    )
    temperature: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        examples=[0.7],
        description=(
            "The temperature to use for sampling the response. "
            "Higher values (e.g., 0.8) make the output more random, "
            "while lower values (e.g., 0.2) make it more focused and deterministic."
        ),
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        examples=[1.0],
        description=(
            "The top-p (nucleus) sampling parameter. "
            "It controls the diversity of the output by limiting the "
            "sampling to the smallest set of tokens whose cumulative probability "
            "is at least `top_p`."
        ),
    )
    max_tokens: int = Field(
        4096,
        ge=1,
        examples=[256],
        description=(
            "The maximum number of tokens to generate in the response. "
            "This limits the length of the output."
        ),
    )
    reasoning: bool = Field(
        False,
        examples=[True],
        description=(
            "Enable the model's extended-thinking / reasoning mode. When true and the "
            "selected model supports it, the model reasons before answering and the "
            "reasoning is streamed on a separate `thinking` SSE channel. Models without "
            "a reasoning mode ignore this flag."
        ),
    )

    @field_validator("messages")
    @classmethod
    def _must_end_with_user(
        cls,
        value: list[ConversationTurn],
    ) -> list[ConversationTurn]:
        if value[-1].role != "user":
            raise ValueError("the last message must be from the user")
        return value

    @field_validator("query_transform_model")
    @classmethod
    def _must_support_query_transform(cls, value: ChatModel) -> ChatModel:
        try:
            return parse_chat_model(value, QUERY_TRANSFORM_MODELS)
        except ValueError as error:
            raise ValueError(
                "query_transform_model must be a chat-capable model",
            ) from error

    @field_validator("inference_model")
    @classmethod
    def _must_be_active_chat_model(cls, value: ChatModel) -> ChatModel:
        try:
            return parse_chat_model(value, CHAT_MODELS)
        except ValueError as error:
            raise ValueError("inference_model must be an active chat model") from error

    @field_validator("embeddings_model")
    @classmethod
    def _must_be_active_embedding_model(cls, value: Model) -> Model:
        if value not in ACTIVE_EMBEDDING_MODELS:
            raise ValueError("embeddings_model must be an active embedding model")
        return value

    @property
    def query(self) -> str:
        """The latest user message; the text used to drive retrieval."""
        return self.messages[-1].content

    @property
    def history(self) -> list[ConversationTurn]:
        """All prior turns, excluding the latest user message."""
        return self.messages[:-1]

    def capped_history(self, max_turns: int) -> list[ConversationTurn]:
        """
        Prior turns limited to the most recent ``max_turns``, guaranteed to start
        with a user turn so the conversation alternates cleanly for strict
        providers (e.g. Anthropic). Returns an empty list when ``max_turns <= 0``.
        """
        if max_turns <= 0:
            return []
        turns = self.history[-max_turns:]
        if turns and turns[0].role == "assistant":
            turns = turns[1:]
        return turns
