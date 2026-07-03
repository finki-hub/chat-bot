import anyio

from app.api import chat_title
from app.llms.models import Model
from app.schemas.chat_title import ChatTitleSchema


def test_chat_title_uses_transform_prompt_and_normalizes_quotes(monkeypatch):
    seen: dict[str, str | int | float | Model] = {}

    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        seen.update(
            {
                "max_tokens": max_tokens,
                "model": model,
                "query": query,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
        return '"Испитна сесија"\n'

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    payload = ChatTitleSchema(
        messages=[{"role": "user", "content": "Кога е јунската сесија?"}],
        query_transform_model=Model.CLAUDE_HAIKU_4_5,
    )

    response = anyio.run(chat_title.generate_chat_title, payload)

    assert response.title == "Испитна сесија"
    assert seen["model"] == Model.CLAUDE_HAIKU_4_5
    assert seen["max_tokens"] == 32
    assert "Кога е јунската сесија?" in str(seen["query"])
    assert "conversation title" in str(seen["system_prompt"])


def test_chat_title_falls_back_to_first_user_message_when_model_returns_empty(
    monkeypatch,
):
    async def fake_transform_query(
        query: str,
        model: Model,
        *,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        return "   "

    monkeypatch.setattr(chat_title, "transform_query", fake_transform_query)
    payload = ChatTitleSchema(
        messages=[
            {
                "role": "user",
                "content": "  Како да пријавам испит?\nИ кои се роковите?  ",
            },
        ],
    )

    response = anyio.run(chat_title.generate_chat_title, payload)

    assert response.title == "Како да пријавам испит?"
