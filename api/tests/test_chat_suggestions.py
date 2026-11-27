from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.api import chat_suggestions
from app.data.connection import Database
from app.llms.models import Model
from app.llms.retrieval_result import RetrievedContext
from app.main import make_app
from app.schemas.chat_suggestions import ChatSuggestionsSchema
from app.utils.settings import Settings
from tests.test_api_endpoints import HealthyDatabase

_SETTINGS = Settings(
    API_KEY="test-api-key",
    CREDENTIAL_ENCRYPTION_KEY="test-credential-key",
    MCP_API_KEY="test-mcp-key",
)
_PAYLOAD = ChatSuggestionsSchema(
    question="Колку чини уписот?",
    answer="Уписот чини 200 денари.",
)
_REQUEST = cast(
    "Request",
    SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=_SETTINGS))),
)
_DB = cast("Database", object())


def test_parse_drops_markers_quotes_duplicates_and_the_question() -> None:
    raw = (
        """1. Како се плаќа уписот?
- „Како се плаќа уписот?“
* Колку чини уписот?

Кој е рокот за упис?
"""
        + "Д" * 101
    )

    assert chat_suggestions.parse_suggestions(raw, "колку чини уписот") == [
        "Како се плаќа уписот?",
        "Кој е рокот за упис?",
    ]


def test_parse_keeps_at_most_five_candidates() -> None:
    raw = "\n".join(f"Прашање {n}?" for n in range(8))

    assert len(chat_suggestions.parse_suggestions(raw, "q")) == 5


def _mock(
    monkeypatch,
    *,
    output: str | None,
    retrieved: dict[str, RetrievedContext] | None = None,
    model: object = Model.GPT_6_LUNA,
) -> list[dict[str, object]]:
    captured: list[dict[str, object]] = []

    async def select(**kwargs):
        return None if model is None else (model, None)

    async def transform(query: str, *args, **kwargs) -> str:
        return query if output is None else output

    async def retrieve(*, query: str, **kwargs) -> RetrievedContext:
        return (retrieved or {}).get(query, RetrievedContext(text=""))

    monkeypatch.setattr(chat_suggestions, "select_helper_model", select)
    monkeypatch.setattr(chat_suggestions, "transform_query", transform)
    monkeypatch.setattr(
        chat_suggestions,
        "get_retrieved_context_with_sources",
        retrieve,
    )
    monkeypatch.setattr(
        chat_suggestions,
        "capture",
        lambda _id, _event, props: captured.append(props),
    )
    return captured


def _strong(text: str = "блок") -> RetrievedContext:
    return RetrievedContext(text=text, top_score=0.7)


@pytest.mark.anyio
async def test_keeps_only_suggestions_the_knowledge_base_can_answer(
    monkeypatch,
) -> None:
    captured = _mock(
        monkeypatch,
        output="Прво?\nСлабо?\nБез резиме?\nВторо?\nТрето?\n",
        retrieved={
            "Прво?": _strong(),
            "Слабо?": RetrievedContext(text="блок", weak=True, top_score=0.01),
            # No score: the reranker failed, so nothing was verified.
            "Без резиме?": RetrievedContext(text="блок"),
            "Второ?": _strong(),
            "Трето?": _strong(),
        },
    )

    response = await chat_suggestions.generate_chat_suggestions(
        _PAYLOAD,
        _DB,
        _REQUEST,
    )

    assert response.suggestions == ["Прво?", "Второ?", "Трето?"]
    assert captured[0]["candidate_count"] == 5
    assert captured[0]["suggestion_count"] == 3


@pytest.mark.anyio
async def test_caps_suggestions_at_three(monkeypatch) -> None:
    lines = [f"Прашање {n}?" for n in range(5)]
    _mock(
        monkeypatch,
        output="\n".join(lines),
        retrieved={line: _strong() for line in lines},
    )

    response = await chat_suggestions.generate_chat_suggestions(
        _PAYLOAD,
        _DB,
        _REQUEST,
    )

    assert response.suggestions == lines[:3]


def test_parse_strips_markdown_emphasis() -> None:
    assert chat_suggestions.parse_suggestions("**Кој е рокот?**\n_Каде?_", "q") == [
        "Кој е рокот?",
        "Каде?",
    ]


@pytest.mark.anyio
async def test_stops_checking_once_three_suggestions_pass(monkeypatch) -> None:
    lines = [f"Прашање {n}?" for n in range(5)]
    _mock(monkeypatch, output="\n".join(lines))
    searched: list[str] = []

    async def retrieve(*, query: str, **kwargs) -> RetrievedContext:
        searched.append(query)
        return _strong()

    monkeypatch.setattr(
        chat_suggestions,
        "get_retrieved_context_with_sources",
        retrieve,
    )

    response = await chat_suggestions.generate_chat_suggestions(
        _PAYLOAD,
        _DB,
        _REQUEST,
    )

    assert response.suggestions == lines[:3]
    assert searched == lines[:3]


@pytest.mark.anyio
async def test_a_failed_search_drops_the_suggestion(monkeypatch) -> None:
    _mock(monkeypatch, output="Прво?")

    async def failing(**kwargs) -> RetrievedContext:
        raise RuntimeError

    monkeypatch.setattr(
        chat_suggestions,
        "get_retrieved_context_with_sources",
        failing,
    )

    response = await chat_suggestions.generate_chat_suggestions(
        _PAYLOAD,
        _DB,
        _REQUEST,
    )

    assert response.suggestions == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("output", "model"),
    [(None, Model.GPT_6_LUNA), ("Прво?", None)],
    ids=["model_failed", "no_model"],
)
async def test_returns_no_suggestions_without_a_working_model(
    monkeypatch,
    output: str | None,
    model: object,
) -> None:
    _mock(
        monkeypatch,
        output=output,
        retrieved={"Прво?": _strong()},
        model=model,
    )

    response = await chat_suggestions.generate_chat_suggestions(
        _PAYLOAD,
        _DB,
        _REQUEST,
    )

    assert response.suggestions == []


def test_endpoint_requires_the_api_key_and_a_valid_conversation_id(
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.main.Database", HealthyDatabase)
    _mock(monkeypatch, output="Прво?", retrieved={"Прво?": _strong()})
    body = {"question": "Колку чини уписот?", "answer": "200 денари."}

    with TestClient(make_app(_SETTINGS)) as client:
        unauthorized = client.post("/chat/suggestions", json=body)
        invalid_id = client.post(
            "/chat/suggestions",
            headers={"x-api-key": "test-api-key", "x-conversation-id": "x"},
            json=body,
        )
        ok = client.post(
            "/chat/suggestions",
            headers={"x-api-key": "test-api-key"},
            json=body,
        )

    assert unauthorized.status_code == 401
    assert invalid_id.status_code == 422
    assert ok.status_code == 200
    assert ok.json() == {"suggestions": ["Прво?"]}
