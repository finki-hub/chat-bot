import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.data.connection import Database
from app.data.questions import get_question_names_query
from app.llms import faq_query_recovery_plan as plan_module
from app.llms.faq_query_recovery_plan import plan_faq_query_recovery
from app.schemas.questions import QuestionSchema


def _question() -> QuestionSchema:
    return QuestionSchema(
        id=uuid4(),
        name="Студентска служба",
        content="Содржина на FAQ. ",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def test_plan_uses_bounded_catalog_and_fetches_one_exact_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _question()
    names = AsyncMock(return_value=[question.name])
    exact = AsyncMock(return_value=question)
    monkeypatch.setattr(plan_module, "get_question_names_query", names)
    monkeypatch.setattr(plan_module, "get_question_by_name_query", exact)
    db = Database("postgresql://unused")

    result = anyio.run(
        plan_faq_query_recovery,
        db,
        "Каде е службата?",
    )

    assert result.outcome == "matched"
    assert result.recovery is not None
    assert result.recovery.title == question.name
    assert result.question == question
    names.assert_awaited_once()
    assert names.await_args is not None
    assert names.await_args.kwargs == {"limit": 513}
    exact.assert_awaited_once_with(db, question.name)


def test_plan_abstains_instead_of_resolving_a_truncated_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = AsyncMock(return_value=[f"FAQ {index}" for index in range(513)])
    exact = AsyncMock()
    monkeypatch.setattr(plan_module, "get_question_names_query", names)
    monkeypatch.setattr(plan_module, "get_question_by_name_query", exact)

    result = anyio.run(
        plan_faq_query_recovery,
        Database("postgresql://unused"),
        "Каде е службата?",
    )

    assert result.outcome == "skipped"
    assert result.recovery is None
    exact.assert_not_awaited()


def test_plan_returns_no_match_for_a_complete_catalog_without_an_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = AsyncMock(return_value=["Деканат", "Компјутерски центар"])
    exact = AsyncMock()
    monkeypatch.setattr(plan_module, "get_question_names_query", names)
    monkeypatch.setattr(plan_module, "get_question_by_name_query", exact)

    result = anyio.run(
        plan_faq_query_recovery,
        Database("postgresql://unused"),
        "Каде е библиотеката?",
    )

    assert result.outcome == "no_match"
    assert result.recovery is None
    exact.assert_not_awaited()


def test_plan_failure_degrades_to_error_outcome_without_query_logging(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail(*args, **kwargs):
        raise RuntimeError("private query details")

    monkeypatch.setattr(plan_module, "get_question_names_query", fail)

    result = anyio.run(
        plan_faq_query_recovery,
        Database("postgresql://unused"),
        "Каде е службата?",
    )

    assert result.outcome == "error"
    assert "private query details" not in caplog.text
    assert "Каде е службата" not in caplog.text


def test_plan_propagates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(plan_module, "get_question_names_query", cancel)

    with pytest.raises(asyncio.CancelledError):
        anyio.run(
            plan_faq_query_recovery,
            Database("postgresql://unused"),
            "Каде е службата?",
        )


def test_ineligible_query_skips_database_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = AsyncMock()
    monkeypatch.setattr(plan_module, "get_question_names_query", names)

    result = anyio.run(
        plan_faq_query_recovery,
        Database("postgresql://unused"),
        "Опиши ја политиката за вработување.",
    )

    assert result.outcome == "skipped"
    names.assert_not_awaited()


def test_title_catalog_limit_is_a_bound_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = Database("postgresql://unused")
    fetch = AsyncMock(return_value=[{"name": "Студентска служба"}])
    monkeypatch.setattr(db, "fetch", fetch)

    async def run() -> list[str]:
        return await get_question_names_query(db, limit=513)

    result = anyio.run(run)

    assert result == ["Студентска служба"]
    assert fetch.await_args is not None
    sql, limit = fetch.await_args.args
    assert "ORDER BY name ASC LIMIT $1" in sql
    assert limit == 513
