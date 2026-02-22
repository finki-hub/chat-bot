from pydantic import BaseModel, Field

from app.constants.defaults import (
    DEFAULT_EMBEDDINGS_MODEL,
    DEFAULT_INFERENCE_MODEL,
)
from app.llms.models import Model


class ChatSchema(BaseModel):
    prompt: str = Field(
        examples=["Where is FINKI located?"],
        description="The user's free-text query to send to the chat system.",
    )
    system_prompt: str | None = Field(
        None,
        examples=["You are a helpful assistant."],
        description=(
            "The system prompt to use for the chat. "
            "This is a fixed prompt that sets the context for the chat model."
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
    inference_model: Model = Field(
        DEFAULT_INFERENCE_MODEL,
        examples=[DEFAULT_INFERENCE_MODEL.value],
        description=(
            "Which Model to use for generating / streaming the response. "
            "Must be one of the values in `app.llms.models.Model`."
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
    rerank_documents: bool = Field(
        True,
        examples=[False],
        description=(
            "Whether to re-rank documents based on their relevance to the query. "
            "If True, the system will re-rank documents before generating a response."
        ),
    )
