from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.llms.models import Model


class StreamRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1, max_length=16_000)
    inference_model: Literal[Model.QWEN3_8B]
    temperature: float = Field(ge=0.1, le=1)
    top_p: float = Field(gt=0, le=1)
    max_tokens: int = Field(ge=1, le=2_048)
    interface: Literal["discord", "web"] = "web"
