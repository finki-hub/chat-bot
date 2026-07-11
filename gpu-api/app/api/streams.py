import logging
from pathlib import Path
from typing import Final

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from app.llms.streams import stream_response
from app.schemas.streams import StreamRequestSchema

logger = logging.getLogger(__name__)

_PROMPTS_DIR: Final = Path(__file__).resolve().parents[2] / "resources" / "prompts"
DEFAULT_SYSTEM_PROMPT: Final = (
    (_PROMPTS_DIR / "agent_system.txt")
    .read_text(
        encoding="utf-8",
    )
    .strip()
)
_FORMAT_PROMPTS: Final = {
    "discord": (_PROMPTS_DIR / "discord_format.txt")
    .read_text(encoding="utf-8")
    .strip(),
    "web": (_PROMPTS_DIR / "web_format.txt").read_text(encoding="utf-8").strip(),
}

router = APIRouter(
    prefix="/stream",
    tags=["Stream"],
)


@router.post(
    "/",
    summary="Stream a chat response from a self-hosted model",
    description="Streams a chat response from a self-hosted model using the configured system prompt.",
    response_model=None,
    status_code=status.HTTP_200_OK,
    operation_id="selfHostedChat",
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "The requested model is not supported.",
        },
    },
)
async def stream(
    payload: StreamRequestSchema,
) -> StreamingResponse:
    logger.info(
        "Received stream request: model=%s prompt_len=%d",
        payload.inference_model.value,
        len(payload.prompt),
    )

    return stream_response(
        user_prompt=payload.prompt,
        model=payload.inference_model,
        system_prompt="\n\n".join(
            [DEFAULT_SYSTEM_PROMPT, _FORMAT_PROMPTS[payload.interface]],
        ),
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
