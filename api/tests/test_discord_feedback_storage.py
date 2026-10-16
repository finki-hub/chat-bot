from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.data.feedback import (
    DiscordFeedbackStaleError,
    DiscordFeedbackUnavailableError,
    upsert_discord_feedback,
)
from app.schemas.feedback import FeedbackSchema
from tests.feedback_test_support import auth_headers, make_client


def _case():
    owner, rater, conversation, message, response = (uuid4() for _ in range(5))
    candidate = {
        "owner_id": owner,
        "rater_id": rater,
        "conversation_id": conversation,
        "id": message,
        "owner_generation": 3,
        "rater_generation": 7,
    }
    conn = SimpleNamespace(
        execute=AsyncMock(return_value="SET"),
        fetch=AsyncMock(
            return_value=[
                {"id": owner, "history_generation": 3},
                {"id": rater, "history_generation": 7},
            ]
        ),
        fetchrow=AsyncMock(
            side_effect=[
                {"id": conversation},
                {
                    "content": "Stored answer",
                    "inference_model": "stored-model",
                    "embeddings_model": "stored-embedding",
                    "query_transform_model": "stored-transform",
                },
                {"id": uuid4(), "response_id": response, "feedback_type": "like"},
            ]
        ),
        fetchval=AsyncMock(return_value="Stored question"),
    )

    @asynccontextmanager
    async def transaction():
        yield conn

    db = SimpleNamespace(
        fetch=AsyncMock(return_value=[candidate]),
        transaction=transaction,
        fetchrow=AsyncMock(return_value={"id": rater, "history_generation": 7}),
    )
    payload = FeedbackSchema(
        response_id=response,
        user_id="222",
        client="discord",
        feedback_type="like",
        question_text="forged",
        answer_text="forged",
        inference_model="forged",
        embeddings_model="forged",
        query_transform_model="forged",
    )
    return db, conn, candidate, payload


@pytest.mark.anyio
async def test_discord_other_rater_uses_stored_context_and_ordered_account_locks():
    db, conn, candidate, payload = _case()
    stored = await upsert_discord_feedback(db, payload)
    assert stored is not None
    assert candidate["owner_id"] != candidate["rater_id"]
    assert stored.feedback.question_text == "Stored question"
    assert stored.feedback.answer_text == "Stored answer"
    assert stored.feedback.inference_model == "stored-model"
    assert stored.feedback.embeddings_model == "stored-embedding"
    assert stored.feedback.query_transform_model == "stored-transform"
    assert conn.fetch.call_args.args[1] == sorted(
        [candidate["owner_id"], candidate["rater_id"]]
    )
    insert_args = conn.fetchrow.call_args_list[-1].args
    assert insert_args[8:13] == (
        "Stored question",
        "Stored answer",
        "stored-model",
        "stored-embedding",
        "stored-transform",
    )
    assert "forged" not in str(insert_args)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reason",
    ["owner_epoch", "rater_epoch", "parent", "message", "missing", "ambiguous"],
)
async def test_feedback_fails_closed_for_stale_missing_or_ambiguous_context(reason):
    db, conn, candidate, payload = _case()
    if reason == "owner_epoch":
        conn.fetch.return_value[0]["history_generation"] += 1
    elif reason == "rater_epoch":
        conn.fetch.return_value[1]["history_generation"] += 1
    elif reason == "parent":
        conn.fetchrow.side_effect = [None]
    elif reason == "message":
        conn.fetchrow.side_effect = [{"id": candidate["conversation_id"]}, None]
    elif reason == "missing":
        db.fetch.return_value = []
    else:
        db.fetch.return_value = [candidate, candidate]
    expected = (
        DiscordFeedbackStaleError
        if reason in {"rater_epoch", "owner_epoch"}
        else DiscordFeedbackUnavailableError
    )
    with pytest.raises(expected):
        await upsert_discord_feedback(db, payload)
    assert not any(
        "INSERT INTO feedback" in call.args[0] for call in conn.fetchrow.call_args_list
    )


def test_discord_feedback_http_missing_response_is_404_not_500():
    db, _, _, payload = _case()
    db.fetch.return_value = []
    client = make_client(db)
    response = client.post(
        "/chat/feedback", headers=auth_headers(), json=payload.model_dump(mode="json")
    )
    assert response.status_code == 404
    assert response.headers["X-Feedback-Generation"] == "7"


def test_discord_feedback_requires_service_authentication():
    db, _, _, payload = _case()
    response = make_client(db).post(
        "/chat/feedback", json=payload.model_dump(mode="json")
    )
    assert response.status_code == 401
    db.fetch.assert_not_awaited()
    db.fetchrow.assert_not_awaited()


@pytest.mark.parametrize("token", ["-1", "+1", "1.5", "", "00", " 7", "٧", "9" * 20])
def test_retry_generation_header_framework_validation(token):
    db, _, _, payload = _case()
    response = make_client(db).post(
        "/chat/feedback",
        headers={**auth_headers(), "X-Feedback-Generation": token.encode("utf-8")},
        json=payload.model_dump(mode="json"),
    )
    assert response.status_code == 422
    db.fetchrow.assert_not_awaited()
    db.fetch.assert_not_awaited()


@pytest.mark.parametrize("with_candidate", [False, True])
@pytest.mark.parametrize("changed", ["rater", "owner", "both"])
def test_first_attempt_stale_snapshot_is_terminal_for_either_captured_account(
    changed, with_candidate
):
    db, conn, _, payload = _case()
    if not with_candidate:
        db.fetch.return_value = []
    if changed in {"rater", "both"}:
        conn.fetch.return_value[1]["history_generation"] = 8
    if changed in {"owner", "both"}:
        conn.fetch.return_value[0]["history_generation"] = 4
    response = make_client(db).post(
        "/chat/feedback", headers=auth_headers(), json=payload.model_dump(mode="json")
    )
    if changed in {"rater", "both"} or with_candidate:
        assert response.status_code == 409
        assert "X-Feedback-Generation" not in response.headers
    else:
        assert response.status_code == 404
        assert response.headers["X-Feedback-Generation"] == "7"
    assert not any(
        "INSERT INTO feedback" in call.args[0] for call in conn.fetchrow.call_args_list
    )


def test_missing_account_has_no_retry_token_and_does_not_lookup_response():
    db, _, _, payload = _case()
    db.fetchrow.return_value = None
    response = make_client(db).post(
        "/chat/feedback", headers=auth_headers(), json=payload.model_dump(mode="json")
    )
    assert response.status_code == 404
    assert "X-Feedback-Generation" not in response.headers
    db.fetch.assert_not_awaited()


def test_generation_is_compared_to_actual_rater_not_owner():
    db, _, _, payload = _case()
    response = make_client(db).post(
        "/chat/feedback",
        headers={**auth_headers(), "X-Feedback-Generation": "3"},
        json=payload.model_dump(mode="json"),
    )
    assert response.status_code == 409


def test_rater_capture_precedes_pending_response_lookup():
    db, _, _, payload = _case()

    async def lookup(*args):
        db.fetchrow.assert_awaited_once()
        return []

    db.fetch.side_effect = lookup
    response = make_client(db).post(
        "/chat/feedback", headers=auth_headers(), json=payload.model_dump(mode="json")
    )
    assert response.status_code == 404
    assert response.headers["X-Feedback-Generation"] == "7"


@pytest.mark.parametrize("missing_account", ["owner", "rater"])
def test_account_removed_after_snapshot_is_terminal_not_500(missing_account):
    db, conn, _, payload = _case()
    conn.fetch.return_value.pop(0 if missing_account == "owner" else 1)
    response = make_client(db).post(
        "/chat/feedback", headers=auth_headers(), json=payload.model_dump(mode="json")
    )
    assert response.status_code == 409
    assert "X-Feedback-Generation" not in response.headers


@pytest.mark.parametrize("generation", ["0", "9" * 19])
def test_bounded_nonnegative_decimal_generations_reach_fence_comparison(generation):
    db, _, _, payload = _case()
    result = make_client(db).post(
        "/chat/feedback",
        headers={**auth_headers(), "X-Feedback-Generation": generation},
        json=payload.model_dump(mode="json"),
    )
    assert result.status_code == 409


def test_openapi_documents_retry_header_terminal_conflict_and_missing_token():
    db, _, _, _ = _case()
    operation = (
        make_client(db).get("/openapi.json").json()["paths"]["/chat/feedback"]["post"]
    )
    header = next(
        parameter
        for parameter in operation["parameters"]
        if parameter["name"] == "X-Feedback-Generation"
    )
    assert header["in"] == "header"
    assert not header["required"]
    assert (
        operation["responses"]["404"]["headers"]["X-Feedback-Generation"]["schema"][
            "maxLength"
        ]
        == 19
    )
    assert "409" in operation["responses"]


def test_pending_first_generation_zero_returns_retry_token_not_missing_account():
    db, conn, _, payload = _case()
    db.fetchrow.return_value["history_generation"] = 0
    conn.fetch.return_value[1]["history_generation"] = 0
    db.fetch.return_value = []
    result = make_client(db).post(
        "/chat/feedback", headers=auth_headers(), json=payload.model_dump(mode="json")
    )
    assert result.status_code == 404
    assert result.headers["X-Feedback-Generation"] == "0"
