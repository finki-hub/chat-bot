from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from app.data.connection import Database
from app.llms import link_context
from app.llms.link_context import get_relevant_links_context

DB = cast("Database", object())
ROWS = [
    {
        "name": "Студентска служба",
        "url": "https://finki.ukim.mk/studentska",
        "description": "Работно време",
    },
    {"name": "Менза", "url": "https://example.com/menza", "description": None},
    {"name": "Распоред", "url": "https://timetables.finki-hub.com", "description": ""},
]


def _install(monkeypatch, scores: list[float]) -> AsyncMock:
    monkeypatch.setattr(
        link_context,
        "fetch_links_for_context",
        AsyncMock(return_value=ROWS),
    )
    response = SimpleNamespace(
        json=lambda: {
            "reranked_documents": [
                {"index": index, "score": score} for index, score in enumerate(scores)
            ],
        },
    )
    rerank = AsyncMock(return_value=response)
    monkeypatch.setattr(link_context, "post_rerank", rerank)
    return rerank


@pytest.mark.anyio
async def test_relevant_links_render_only_links_above_the_score_floor(
    monkeypatch,
) -> None:
    rerank = _install(monkeypatch, [0.9, 0.0, 0.5])

    rendered = await get_relevant_links_context(DB, "работно време")

    assert rendered == (
        "Корисни линкови:\n"
        "- Студентска служба: https://finki.ukim.mk/studentska (Работно време)\n"
        "- Распоред: https://timetables.finki-hub.com"
    )
    assert rerank.await_args is not None
    assert rerank.await_args.args[0]["query"] == "работно време"
    assert len(rerank.await_args.args[0]["documents"]) == len(ROWS)


@pytest.mark.anyio
async def test_no_relevant_links_render_nothing(monkeypatch) -> None:
    _install(monkeypatch, [0.0, 0.0, 0.0])

    assert await get_relevant_links_context(DB, "менза") == ""


@pytest.mark.anyio
async def test_link_catalog_failure_renders_nothing(monkeypatch) -> None:
    rerank = _install(monkeypatch, [0.9, 0.9, 0.9])
    monkeypatch.setattr(
        link_context,
        "fetch_links_for_context",
        AsyncMock(side_effect=OSError("db down")),
    )

    assert await get_relevant_links_context(DB, "менза") == ""
    rerank.assert_not_awaited()


@pytest.mark.anyio
async def test_link_rerank_failure_renders_nothing(monkeypatch) -> None:
    rerank = _install(monkeypatch, [])
    rerank.side_effect = OSError("reranker down")

    assert await get_relevant_links_context(DB, "менза") == ""
