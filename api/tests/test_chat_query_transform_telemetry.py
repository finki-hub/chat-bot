from uuid import uuid4

import pytest

from app.api import chat as chat_api
from app.llms.agents import StreamObservation
from app.llms.query_modes import QueryTransformMode
from app.schemas.chat import ChatSchema
from app.utils.timing import RequestTimings


def test_generation_telemetry_distinguishes_requested_and_effective_transform_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = []

    def capture(distinct_id, event, properties):
        captured.append((distinct_id, event, properties))

    monkeypatch.setattr(chat_api, "capture", capture)
    response_id = uuid4()
    payload = ChatSchema.model_validate(
        {
            "messages": [{"role": "user", "content": "query"}],
            "query_transform_mode": QueryTransformMode.REWRITE_HYDE,
        },
    )

    chat_api._capture_chat_response(  # noqa: SLF001
        distinct_id="user",
        payload=payload,
        response_id=response_id,
        timings=RequestTimings(),
        retrieval_hit=False,
        usage=None,
        outcome="empty_answer",
        observation=StreamObservation(
            distinct_id="user",
            response_id=str(response_id),
        ),
        answer_text="",
        session_id=None,
        effective_transform_mode=QueryTransformMode.RAW,
    )

    properties = captured[0][2]
    assert properties["requested_query_transform_mode"] == "rewrite_hyde"
    assert properties["query_transform_mode"] == "raw"
