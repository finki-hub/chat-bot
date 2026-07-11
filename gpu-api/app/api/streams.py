from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Final

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.llms.qwen3 import stream_qwen3_response
from app.schemas.streams import StreamRequestSchema

_PROMPTS_DIR: Final = Path(__file__).resolve().parents[2] / "resources" / "prompts"
_SYSTEM_PROMPT: Final = (
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

router = APIRouter(prefix="/stream", tags=["Stream"])


@router.post("/", response_class=EventSourceResponse)
async def stream(payload: StreamRequestSchema) -> AsyncGenerator[ServerSentEvent]:
    async for token in stream_qwen3_response(
        payload.prompt,
        "\n\n".join((_SYSTEM_PROMPT, _FORMAT_PROMPTS[payload.interface])),
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    ):
        yield ServerSentEvent(raw_data=token)
