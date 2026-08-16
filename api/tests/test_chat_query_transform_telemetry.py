from uuid import uuid4

import pytest

from app.api import chat as chat_api
from app.llms.agents import StreamObservation
from app.llms.query_modes import QueryTransformMode
from app.schemas.chat import ChatSchema
from app.utils.timing import RequestTimings


def _capture_properties(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    timings: RequestTimings | None = None,
):
    captured = []

    def capture(distinct_id, event, properties):
        captured.append((distinct_id, event, properties))

    monkeypatch.setattr(chat_api, "capture", capture)
    response_id = uuid4()
    payload = ChatSchema.model_validate(
        {
            "messages": [{"role": "user", "content": query}],
            "query_transform_mode": QueryTransformMode.REWRITE_HYDE,
        },
    )

    chat_api._capture_chat_response(  # noqa: SLF001
        distinct_id="user",
        payload=payload,
        response_id=response_id,
        timings=timings or RequestTimings(),
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

    return captured[0][2]


def test_generation_telemetry_distinguishes_requested_and_effective_transform_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = _capture_properties(monkeypatch, "query")

    assert properties["requested_query_transform_mode"] == "rewrite_hyde"
    assert properties["query_transform_mode"] == "raw"


@pytest.mark.parametrize(
    ("query", "expected_script"),
    [
        ("smerovi na finki", "latin"),
        ("š č ž ǵ ḱ", "latin"),
        ("смерови на финки", "cyrillic"),
        ("smerovi на finki", "mixed"),
        ("12345?!", "other"),
    ],
)
def test_generation_telemetry_records_query_script(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expected_script: str,
) -> None:
    properties = _capture_properties(monkeypatch, query)

    assert properties["query_script"] == expected_script


def test_generation_telemetry_emits_content_free_retrieval_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timings = RequestTimings()
    timings.transliteration_variant_added = True
    timings.lexical_search_outcome = "matched"
    timings.dense_faq_candidate_count = 1
    timings.dense_document_candidate_count = 2
    timings.lexical_faq_candidate_count = 3
    timings.final_faq_count = 1
    timings.final_document_count = 2
    timings.lexical_only_final_count = 1
    timings.retrieval_path = "hybrid"
    timings.reranker_fallback = False
    private_query = "sakam privaten odgovor"

    properties = _capture_properties(monkeypatch, private_query, timings)

    assert properties["transliteration_variant_added"] is True
    assert properties["lexical_search_outcome"] == "matched"
    assert properties["dense_faq_candidate_count"] == 1
    assert properties["dense_document_candidate_count"] == 2
    assert properties["lexical_faq_candidate_count"] == 3
    assert properties["final_faq_count"] == 1
    assert properties["final_document_count"] == 2
    assert properties["lexical_only_final_count"] == 1
    assert properties["retrieval_path"] == "hybrid"
    assert properties["reranker_fallback"] is False
    assert private_query not in repr(properties)
