from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llms.models import Model


class StreamRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(description="The user's prompt.")
    inference_model: Model = Field(
        description="The inference model to use for the chat response.",
    )
    temperature: float = Field(
        ge=0,
        le=1,
        description="The temperature for the model's response generation.",
    )
    top_p: float = Field(
        ge=0,
        le=1,
        description="The top-p value for nucleus sampling in response generation.",
    )
    max_tokens: int = Field(
        ge=1,
        le=4096,
        description="The maximum number of tokens to generate in the response.",
    )
    interface: Literal["discord", "web"] = Field(
        default="web",
        description="The response formatting profile.",
    )
