from fastapi import APIRouter, status

from app.llms.query_transform import transform_query
from app.schemas.chat_title import ChatTitleResponse, ChatTitleSchema

router = APIRouter(prefix="/chat", tags=["Chat"])

_TITLE_MAX = 60
_FALLBACK_TITLE = "Нов разговор"
_TITLE_SYSTEM_PROMPT = """Generate a concise conversation title.

Rules:
- Use the same language as the user's first message.
- Prefer Macedonian Cyrillic when the message is Macedonian.
- Output only the title, with no quotes, punctuation wrapper, markdown, or explanation.
- Keep it under 6 words and under 60 characters.
- Describe the user's intent, not the assistant response."""


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _clip_title(text: str) -> str:
    if len(text) <= _TITLE_MAX:
        return text
    return f"{text[: _TITLE_MAX - 1].rstrip()}…"


def _normalize_title(raw_title: str, fallback_text: str) -> str:
    title = " ".join(_first_line(raw_title).strip("'\"`“”‘’ ").split())
    if not title:
        title = _first_line(fallback_text) or _FALLBACK_TITLE
    return _clip_title(title)


async def generate_chat_title(payload: ChatTitleSchema) -> ChatTitleResponse:
    prompt = f"Conversation transcript:\n{payload.transcript}"
    raw_title = await transform_query(
        prompt,
        payload.query_transform_model,
        system_prompt=_TITLE_SYSTEM_PROMPT,
        temperature=0.2,
        top_p=1.0,
        max_tokens=32,
    )
    return ChatTitleResponse(
        title=_normalize_title(raw_title, payload.first_user_text),
    )


@router.post(
    "/title",
    summary="Generate a chat title",
    description="Generate a short title from the first conversation turns.",
    operation_id="generateChatTitle",
    status_code=status.HTTP_200_OK,
)
async def chat_title(payload: ChatTitleSchema) -> ChatTitleResponse:
    return await generate_chat_title(payload)
