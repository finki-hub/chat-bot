import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import anyio
import pytest

from app.data.connection import Database
from app.llms import context as context_module
from app.llms import faq_evidence_constraints as constraints_module
from app.llms import faq_query_recovery_plan as recovery_plan_module
from app.llms.faq_evidence_constraints import (
    FaqConstraints,
    apply_faq_constraints_to_candidates,
    parse_explicit_faq_constraints,
    plan_faq_evidence_constraints,
)
from app.llms.faq_query_recovery import (
    faq_query_entity_forms,
    is_faq_query_recovery_eligible,
)
from app.llms.models import Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import QueryVariant, QueryVariantBundle
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from app.utils.timing import (
    record_faq_evidence_constraint,
    reset_request_timings,
    start_request_timings,
)

_OFFICES = (
    "Студентска служба",
    "Деканат",
    "Компјутерски центар",
    "Архива",
    "Контакт и локација",
)


def _faq(title: str, *, content: str | None = None) -> QuestionSchema:
    return QuestionSchema(
        id=uuid4(),
        name=title,
        content=content or f"Содржина за {title}.",
        links={},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _install_pipeline_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: QueryTransformMode,
    candidates: list[QuestionSchema],
    rerank: object,
) -> list[str]:
    planned_queries: list[str] = []

    async def contextualize(query, *args, **kwargs):
        return query

    async def variants(query, *args, **kwargs):
        requested = args[1] if len(args) > 1 else kwargs.get("mode", mode)
        variants_list: tuple[QueryVariant, ...]
        if requested == QueryTransformMode.RAW:
            variants_list = (QueryVariant(kind="raw", text=query, is_document=False),)
        elif requested == QueryTransformMode.REWRITE:
            variants_list = (
                QueryVariant(
                    kind="rewrite",
                    text="rewrite without user constraints",
                    is_document=False,
                ),
                QueryVariant(kind="raw", text=query, is_document=False),
            )
        elif requested == QueryTransformMode.HYDE:
            variants_list = (
                QueryVariant(
                    kind="hyde", text="hyde without user constraints", is_document=True
                ),
                QueryVariant(kind="raw", text=query, is_document=False),
            )
        else:
            variants_list = (
                QueryVariant(
                    kind="rewrite",
                    text="rewrite without user constraints",
                    is_document=False,
                ),
                QueryVariant(
                    kind="hyde", text="hyde without user constraints", is_document=True
                ),
                QueryVariant(kind="raw", text=query, is_document=False),
            )
        return QueryVariantBundle(
            variants=variants_list, rerank_query=variants_list[0].text
        )

    async def embed(*args, **kwargs):
        return [1.0]

    async def search(*args, **kwargs):
        return candidates, []

    async def plan(db, query):
        planned_queries.append(query)
        return await constraints_module.plan_faq_evidence_constraints(db, query)

    async def no_lexical(*args, **kwargs):
        return []

    monkeypatch.setattr(context_module, "has_provider_credential", lambda *args: True)
    monkeypatch.setattr(context_module, "_contextualize_query", contextualize)
    monkeypatch.setattr(context_module, "build_query_variants", variants)
    monkeypatch.setattr(context_module, "_embed_variant", embed)
    monkeypatch.setattr(context_module, "_search_both", search)
    monkeypatch.setattr(context_module, "get_matching_questions", no_lexical)
    monkeypatch.setattr(context_module, "plan_faq_evidence_constraints", plan)
    monkeypatch.setattr(context_module, "_post_rerank", rerank)
    return planned_queries


@pytest.mark.parametrize("mode", list(QueryTransformMode))
def test_original_institution_constraint_survives_all_transform_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: QueryTransformMode,
) -> None:
    query = "Каде е студентската служба на Правен факултет?"
    candidates = [_faq("Студентска служба"), _faq("Контакт и локација")]
    names = AsyncMock(return_value=list(_OFFICES))
    monkeypatch.setattr(constraints_module, "get_question_names_query", names)
    rerank = AsyncMock(
        side_effect=AssertionError("all mapped office FAQs are filtered")
    )
    planned_queries = _install_pipeline_mocks(
        monkeypatch,
        mode=mode,
        candidates=candidates,
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=mode,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    assert result.text == ""
    assert planned_queries == [query]
    assert timings.faq_evidence_constraint_applied is True
    assert timings.faq_evidence_constraint_reason == "institution_scope"
    assert timings.faq_evidence_excluded_candidate_count == 2
    assert timings.faq_evidence_constrained_abstained is True
    rerank.assert_not_awaited()


def test_original_exclusion_filters_exact_faq_added_from_contextualized_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Каде е Деканатот? Не прашувам за студентската служба."
    contextualized = "Каде е службата?"
    student_service = _faq("Студентска служба")
    monkeypatch.setattr(
        constraints_module,
        "get_question_names_query",
        AsyncMock(return_value=list(_OFFICES)),
    )
    exact = AsyncMock(return_value=student_service)
    monkeypatch.setattr(recovery_plan_module, "get_question_by_name_query", exact)
    rerank = AsyncMock(side_effect=AssertionError("excluded exact FAQ must not rerank"))
    planned_queries = _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.RAW,
        candidates=[],
        rerank=rerank,
    )
    monkeypatch.setattr(
        context_module,
        "_contextualize_query",
        AsyncMock(return_value=contextualized),
    )
    db = Database("postgresql://unused")

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            db,
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
            history_text="prior user context",
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    assert result.text == ""
    assert planned_queries == [query]
    exact.assert_awaited_once_with(db, "Студентска служба")
    assert timings.local_faq_recovery_outcome == "matched"
    assert timings.faq_evidence_constraint_reason == "entity_exclusion"
    assert timings.faq_evidence_excluded_candidate_count == 1
    assert timings.faq_evidence_constrained_abstained is True
    rerank.assert_not_awaited()


@pytest.mark.parametrize("secondary_title_collision", [False, True])
def test_exclusion_constraint_survives_unrelated_office_title_collision(
    monkeypatch: pytest.MonkeyPatch,
    secondary_title_collision: bool,
) -> None:
    query = "Каде е Деканатот, не прашувам за студентската служба?"
    student_service = _faq("Студентска служба")
    deanery = _faq("Деканат")
    catalog = (
        [*_OFFICES, "Kontakt i lokacija"]
        if secondary_title_collision
        else list(_OFFICES)
    )
    monkeypatch.setattr(
        constraints_module,
        "get_question_names_query",
        AsyncMock(return_value=catalog),
    )
    rerank_payloads: list[dict[str, object]] = []

    class Response:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        rerank_payloads.append(payload)
        return Response()

    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=[student_service, deanery],
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    assert student_service.content not in result.text
    if secondary_title_collision:
        assert result.text == ""
        assert result.sources == ()
        assert rerank_payloads == []
        assert timings.faq_evidence_constraint_reason == "catalog_error"
    else:
        assert deanery.content in result.text
        assert [source.title for source in result.sources] == ["Деканат"]
        assert len(rerank_payloads) == 1
        assert rerank_payloads[0]["documents"] == [
            f"Наслов: Деканат\nСодржина: {deanery.content}"
        ]
        assert timings.faq_evidence_constraint_reason == "entity_exclusion"
        assert timings.faq_evidence_excluded_candidate_count == 1


def test_exclusion_ignores_unrelated_office_collision_when_fact_source_is_unique() -> (
    None
):
    constraints = parse_explicit_faq_constraints(
        "Каде е деканатот, не прашувам за студентската служба?",
        [*_OFFICES, "Компјутерски центар"],
    )

    assert constraints is not None
    assert constraints.kind == "exclusion"
    assert constraints.excluded_title == "Студентска служба"
    assert constraints.direct_fact_source_titles == ("Контакт и локација",)


@pytest.mark.parametrize(
    ("query", "candidates", "expected_titles", "forbidden_text"),
    [
        (
            "Не прашувам за студентската служба, каде е деканатот на ФИНКИ?",
            [
                _faq(
                    "Студентска служба",
                    content="Телефон на студентска служба: 070 302 440.",
                ),
                _faq("Деканат", content="Деканатот е во административната зграда."),
                _faq(
                    "Контакт и локација",
                    content=(
                        "02 370 377, Декан; 070 302 440, Студентска служба; "
                        "contact@finki.ukim.mk"
                    ),
                ),
            ],
            ["Деканат"],
            "070 302 440",
        ),
        (
            "Не прашувам за деканатот, каде е архивата?",
            [
                _faq(
                    "Архива",
                    content=(
                        "Архивата е во Деканатот, горе по скалите; arhiva@finki.ukim.mk"
                    ),
                ),
                _faq(
                    "Контакт и локација",
                    content="02 370 377, Декан; 070 302 440, Студентска служба.",
                ),
            ],
            ["Архива"],
            "070 302 440",
        ),
    ],
)
def test_exclusion_withholds_mapped_direct_facts_without_dropping_alternative_faqs(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    candidates: list[QuestionSchema],
    expected_titles: list[str],
    forbidden_text: str,
) -> None:
    monkeypatch.setattr(
        constraints_module,
        "get_question_names_query",
        AsyncMock(return_value=list(_OFFICES)),
    )

    class Response:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    rerank_payloads: list[dict[str, object]] = []

    async def rerank(payload):
        rerank_payloads.append(payload)
        return Response()

    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=candidates,
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    result = anyio.run(run)

    assert [source.title for source in result.sources] == expected_titles
    assert forbidden_text not in result.text
    assert len(rerank_payloads) == 1
    documents = rerank_payloads[0]["documents"]
    assert isinstance(documents, list)
    assert [
        line.split("\n", 1)[0].removeprefix("Наслов: ") for line in documents
    ] == expected_titles


@pytest.mark.parametrize("rerank_result", ["low", "invalid", "error"])
def test_constrained_queries_never_use_unscored_or_score_floor_fallback(
    monkeypatch: pytest.MonkeyPatch,
    rerank_result: str,
) -> None:
    query = "Каде е студентската служба на Правен факултет?"
    faq = _faq("Општи информации за упис")
    monkeypatch.setattr(
        constraints_module,
        "get_question_names_query",
        AsyncMock(return_value=list(_OFFICES)),
    )

    class Response:
        def json(self):
            if rerank_result == "low":
                return {"reranked_documents": [{"index": 0, "score": 0.01}]}
            return {"reranked_documents": []}

    async def rerank(_payload):
        if rerank_result == "error":
            raise TimeoutError("reranker unavailable")
        return Response()

    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=[_faq("Студентска служба"), faq],
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    assert result.text == ""
    assert faq.content not in result.text
    assert timings.faq_evidence_constrained_abstained is True
    assert timings.faq_evidence_excluded_candidate_count == 1


@pytest.mark.parametrize("rerank_result", ["low", "invalid", "error"])
def test_finki_scope_keeps_ordinary_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    rerank_result: str,
) -> None:
    query = "Каде е студентската служба на ФИНКИ?"
    office = _faq(
        "Студентска служба",
        content="Студентската служба е во ТМФ.",
    ).model_copy(update={"distance": 0.2})
    monkeypatch.setattr(
        constraints_module,
        "get_question_names_query",
        AsyncMock(return_value=list(_OFFICES)),
    )

    class Response:
        def json(self):
            if rerank_result == "low":
                return {"reranked_documents": [{"index": 0, "score": 0.01}]}
            return {"reranked_documents": []}

    async def rerank(_payload):
        if rerank_result == "error":
            raise TimeoutError("reranker unavailable")
        return Response()

    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=[office],
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    assert office.content in result.text
    assert timings.faq_evidence_constraint_reason == "institution_scope"
    assert timings.faq_evidence_excluded_candidate_count == 0
    assert timings.faq_evidence_constrained_abstained is False


def test_constraint_catalog_error_fails_closed_and_cancels_owned_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Каде е студентската служба на Правен факултет?"
    embedding_cancelled = asyncio.Event()

    async def failed_catalog(*args, **kwargs):
        await asyncio.sleep(0)
        raise TimeoutError("bounded FAQ catalog unavailable")

    async def blocked_embedding(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            embedding_cancelled.set()

    monkeypatch.setattr(constraints_module, "get_question_names_query", failed_catalog)
    monkeypatch.setattr(context_module, "has_provider_credential", lambda *args: True)
    monkeypatch.setattr(
        context_module, "_contextualize_query", AsyncMock(return_value=query)
    )
    monkeypatch.setattr(
        context_module,
        "build_query_variants",
        AsyncMock(
            return_value=QueryVariantBundle(
                variants=(QueryVariant(kind="raw", text=query, is_document=False),),
                rerank_query=query,
            ),
        ),
    )
    monkeypatch.setattr(context_module, "_embed_variant", blocked_embedding)

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.RAW,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    assert result.text == ""
    assert embedding_cancelled.is_set()
    assert timings.faq_evidence_constraint_reason == "catalog_error"
    assert timings.faq_evidence_constrained_abstained is True


def test_constraint_planning_cancellation_drains_owned_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Каде е студентската служба на Правен факултет?"
    planner_started = asyncio.Event()
    embedding_started = asyncio.Event()
    embedding_cleaned = asyncio.Event()

    async def blocked_catalog(*args, **kwargs):
        planner_started.set()
        await asyncio.Event().wait()

    async def blocked_embedding(*args, **kwargs):
        embedding_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            embedding_cleaned.set()

    monkeypatch.setattr(constraints_module, "get_question_names_query", blocked_catalog)
    monkeypatch.setattr(context_module, "has_provider_credential", lambda *args: True)
    monkeypatch.setattr(
        context_module, "_contextualize_query", AsyncMock(return_value=query)
    )
    monkeypatch.setattr(
        context_module,
        "build_query_variants",
        AsyncMock(
            return_value=QueryVariantBundle(
                variants=(QueryVariant(kind="raw", text=query, is_document=False),),
                rerank_query=query,
            ),
        ),
    )
    monkeypatch.setattr(context_module, "_embed_variant", blocked_embedding)

    async def run():
        task = asyncio.create_task(
            context_module.get_retrieved_context_with_sources(
                Database("postgresql://unused"),
                query,
                Model.BGE_M3_LOCAL,
                Model.GPT_5_4_MINI,
                query_transform_mode=QueryTransformMode.RAW,
            ),
        )
        await planner_started.wait()
        await embedding_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    anyio.run(run)
    assert embedding_cleaned.is_set()


@pytest.mark.parametrize(
    "query",
    [
        "Каде е студентската служба на Правен факултет?",
        "Kade e studentska sluzba na Praven fakultet?",
        "Каде е студентска служба на Факултет за медицина?",
        "Каде е Деканатот на непознат универзитет?",
    ],
)
def test_foreign_or_unknown_institution_scope_is_recognized(query: str) -> None:
    constraints = parse_explicit_faq_constraints(query, _OFFICES)

    assert constraints is not None
    assert constraints.kind == "institution"
    assert constraints.institution_is_finki is False
    assert constraints.reason == "institution_scope"
    assert constraints.target_title in {"Студентска служба", "Деканат"}


def test_parser_uses_reviewed_title_set_when_catalog_is_not_supplied() -> None:
    constraints = parse_explicit_faq_constraints(
        "Каде е студентската служба на Правен факултет?",
    )

    assert constraints is not None
    assert constraints.target_title == "Студентска служба"


@pytest.mark.parametrize(
    ("query", "kind", "target", "finki", "excluded"),
    [
        (
            "Кога работи Архива на Медицински факултет?",
            "institution",
            "Архива",
            False,
            None,
        ),
        (
            "Кој е телефонот на Деканатот на Универзитет Нова Академија?",
            "institution",
            "Деканат",
            False,
            None,
        ),
        (
            "Кој е мејлот на Компјутерскиот центар на Економски факултет?",
            "institution",
            "Компјутерски центар",
            False,
            None,
        ),
        (
            "Не прашувам за студентската служба. Што значи зборот служба?",
            "exclusion",
            None,
            None,
            "Студентска служба",
        ),
        (
            "Ne prashuvam za studentska sluzba. Sto znaci zborot sluzba?",
            "exclusion",
            None,
            None,
            "Студентска служба",
        ),
        (
            "Не прашувам за студентската служба, каде е деканатот на ФИНКИ?",
            "exclusion",
            "Деканат",
            True,
            "Студентска служба",
        ),
        (
            "Не прашувам за деканатот, каде е архивата?",
            "exclusion",
            None,
            None,
            "Деканат",
        ),
        (
            "Што значи зборот служба? Не прашувам за студентската служба.",
            "exclusion",
            None,
            None,
            "Студентска служба",
        ),
    ],
)
def test_reviewed_scope_and_prefix_exclusion_forms(
    query: str,
    kind: str,
    target: str | None,
    finki: bool | None,
    excluded: str | None,
) -> None:
    constraints = parse_explicit_faq_constraints(query, _OFFICES)

    assert constraints is not None
    assert constraints.kind == kind
    assert constraints.target_title == target
    assert constraints.institution_is_finki is finki
    assert constraints.excluded_title == excluded


@pytest.mark.parametrize(
    "institution",
    ["Медицински факултет", "ФИНКИ"],
)
def test_institution_contact_email_maps_only_to_reviewed_contact_faq(
    institution: str,
) -> None:
    constraints = parse_explicit_faq_constraints(
        f"Која е контакт е-поштата на {institution}?",
        _OFFICES,
    )

    assert constraints is not None
    assert constraints.kind == "institution"
    assert constraints.target_title == "Контакт и локација"
    assert constraints.institution_is_finki is (institution == "ФИНКИ")


@pytest.mark.parametrize(
    "query",
    [
        "Која е контакт е-поштата на Професор Иванов?",
        "Која е контакт е-поштата на професорот од Правен факултет?",
        "Која е контакт е-поштата на Правен факултет и ФИНКИ?",
        "Како да контактирам со Медицински факултет?",
    ],
)
def test_unreviewed_person_or_generic_contact_queries_are_not_scoped(
    query: str,
) -> None:
    assert parse_explicit_faq_constraints(query, _OFFICES) is None


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        ("Каде е студентската служба на Правен факултет?", "Студентска служба"),
        ("Каде е Деканатот на Правен факултет?", "Деканат"),
        (
            "Каде е Компјутерскиот центар на Правен факултет?",
            "Компјутерски центар",
        ),
        ("Каде е архивата на Правен факултет?", "Архива"),
        (
            "Каде е Контакт и локација на Правен факултет?",
            "Контакт и локација",
        ),
        (
            "Кој е телефонскиот број на Деканатот на ФИНКИ?",
            "Деканат",
        ),
    ],
)
def test_all_reviewed_office_faq_entities_resolve_in_qualified_attributes(
    query: str,
    expected_title: str,
) -> None:
    constraints = parse_explicit_faq_constraints(query, _OFFICES)

    assert constraints is not None
    assert constraints.target_title == expected_title


@pytest.mark.parametrize(
    "query",
    [
        "Каде е студентската служба на ФИНКИ денес?",
        "Каде е студентската служба на ФИНКИ во понеделник?",
        (
            "Каде е студентската служба на Факултетот за информатички науки "
            "и компјутерско инженерство денес?"
        ),
        (
            "Каде е студентската служба на Факултетот за информатички науки "
            "и компјутерско инженерство во Скопје?"
        ),
    ],
)
def test_finki_dates_and_trailing_modifiers_do_not_become_foreign_scope(
    query: str,
) -> None:
    assert parse_explicit_faq_constraints(query, _OFFICES) is None


@pytest.mark.parametrize(
    ("query", "expected_title"),
    [
        (
            (
                "Каде се наоѓа студентската служба на Факултетот за информатички "
                "науки и компјутерско инженерство?"
            ),
            "Студентска служба",
        ),
        (
            (
                "Кое е работното време на Студентската служба на Faculty of "
                "Computer Science and Engineering?"
            ),
            "Студентска служба",
        ),
        (
            (
                "Кој е телефонскиот број на Деканатот на Faculty of Computer "
                "Science and Engineering?"
            ),
            "Деканат",
        ),
    ],
)
def test_extended_constraint_grammar_accepts_reviewed_full_finki_names(
    query: str,
    expected_title: str,
) -> None:
    assert not is_faq_query_recovery_eligible(query)
    assert faq_query_entity_forms(query) == ()
    constraints = parse_explicit_faq_constraints(query, _OFFICES)
    assert constraints is not None
    assert constraints.institution_is_finki is True
    assert constraints.target_title == expected_title


@pytest.mark.parametrize(
    "institution",
    [
        "ФИНКИ",
        "FINKI",
        "Факултетот за информатички науки и компјутерско инженерство",
        "Faculty of Computer Science and Engineering",
    ],
)
def test_reviewed_finki_qualifiers_are_compatible(institution: str) -> None:
    constraints = parse_explicit_faq_constraints(
        f"Каде е студентската служба на {institution}?",
        _OFFICES,
    )

    assert constraints is not None
    assert constraints.institution_is_finki is True


@pytest.mark.parametrize(
    "query",
    [
        "Каде е Деканатот? Не прашувам за студентската служба.",
        "Што значи зборот 'служба'? Не прашувам за студентската служба.",
        "Што значи зборот „служба“? Не прашувам за студентската служба.",
        "Sto znaci zborot 'sluzba'? Ne prasuvam za studentskata sluzba.",
        "What does service mean? Не прашувам за студентската служба.",
    ],
)
def test_explicit_request_exclusion_requires_reviewed_complete_head(query: str) -> None:
    constraints = parse_explicit_faq_constraints(query, _OFFICES)

    assert constraints is not None
    assert constraints.kind == "exclusion"
    assert constraints.excluded_title == "Студентска служба"


@pytest.mark.parametrize(
    "query",
    [
        "Што значи зборот 'служба'?",
        "Каде е 'студентска служба'?",
        "Што значи реченицата „не прашувам за студентската служба“?",
        'Што значи реченицата "не прашувам за студентската служба"?',
        "Што значи реченицата „не прашувам за студентската служба?",
        "Не работи студентската служба?",
        "Дали може да се префрлам од Правен на ФИНКИ?",
        "Каде е студентската служба на Правен факултет и ФИНКИ?",
        "Каде е Деканатот? Не прашувам за Архива и Студентска служба.",
    ],
)
def test_quotes_negation_comparison_and_complex_exclusion_do_not_trigger(
    query: str,
) -> None:
    assert parse_explicit_faq_constraints(query, _OFFICES) is None


def test_catalog_collision_abstains_instead_of_resolving_alias() -> None:
    assert (
        parse_explicit_faq_constraints(
            "Каде е службата на Правен факултет?",
            [*_OFFICES, "Службата"],
        )
        is None
    )
    assert (
        parse_explicit_faq_constraints(
            "Каде е студентската служба на Правен факултет?",
            [*_OFFICES, "Kontakt i lokacija"],
        )
        is None
    )


def test_foreign_constraint_removes_only_mapped_office_faqs() -> None:
    office_titles = ("Студентска служба", "Деканат", "Контакт и локација")
    candidates = [
        SimpleNamespace(
            source="faq",
            retrieval_source=SimpleNamespace(title="Студентска служба"),
        ),
        SimpleNamespace(
            source="faq",
            retrieval_source=SimpleNamespace(title="Контакт и локација"),
        ),
        SimpleNamespace(source="faq", retrieval_source=SimpleNamespace(title="Уписи")),
        SimpleNamespace(
            source="chunk", retrieval_source=SimpleNamespace(title="Прием")
        ),
    ]
    constraints = FaqConstraints(
        kind="institution",
        reason="institution_scope",
        office_titles=office_titles,
        institution_is_finki=False,
        target_title="Студентска служба",
    )

    remaining = apply_faq_constraints_to_candidates(candidates, constraints)

    assert remaining == candidates[2:]


def test_exclusion_keeps_alternative_office_and_document_candidates() -> None:
    candidates = [
        SimpleNamespace(
            source="faq",
            retrieval_source=SimpleNamespace(title="Студентска служба"),
        ),
        SimpleNamespace(
            source="faq",
            retrieval_source=SimpleNamespace(title="Деканат"),
        ),
        SimpleNamespace(
            source="chunk", retrieval_source=SimpleNamespace(title="Прием")
        ),
    ]
    constraints = FaqConstraints(
        kind="exclusion",
        reason="entity_exclusion",
        office_titles=_OFFICES,
        excluded_title="Студентска служба",
    )

    remaining = apply_faq_constraints_to_candidates(candidates, constraints)

    assert remaining == candidates[1:]


def test_finki_institution_constraint_preserves_mapped_office_faqs() -> None:
    candidates = [
        SimpleNamespace(
            source="faq",
            retrieval_source=SimpleNamespace(title="Студентска служба"),
        ),
    ]
    constraints = FaqConstraints(
        kind="institution",
        reason="institution_scope",
        office_titles=_OFFICES,
        institution_is_finki=True,
    )

    assert apply_faq_constraints_to_candidates(candidates, constraints) == candidates


def test_later_abstention_diagnostics_preserve_filtered_count_and_reset_per_request() -> (
    None
):
    timings, token = start_request_timings()
    try:
        record_faq_evidence_constraint(
            applied=True,
            reason="institution_scope",
            excluded_candidate_count=1,
        )
        record_faq_evidence_constraint(
            applied=True,
            reason="institution_scope",
            constrained_abstained=True,
        )
        assert timings.faq_evidence_excluded_candidate_count == 1
        assert timings.faq_evidence_constrained_abstained is True
    finally:
        reset_request_timings(token)

    next_timings, next_token = start_request_timings()
    try:
        assert next_timings.faq_evidence_excluded_candidate_count == 0
        assert next_timings.faq_evidence_constraint_applied is False
    finally:
        reset_request_timings(next_token)


@pytest.mark.parametrize(
    ("titles", "expected_outcome"),
    [
        ([*_OFFICES, "Службата"], "error"),
        ([*_OFFICES, "Kontakt i lokacija"], "error"),
        ([f"FAQ {index}" for index in range(513)], "error"),
    ],
)
def test_planner_fails_closed_on_title_collision_or_catalog_overflow(
    monkeypatch: pytest.MonkeyPatch,
    titles: list[str],
    expected_outcome: str,
) -> None:
    monkeypatch.setattr(
        constraints_module,
        "get_question_names_query",
        AsyncMock(return_value=titles),
    )

    result = anyio.run(
        plan_faq_evidence_constraints,
        Database("postgresql://unused"),
        "Каде е службата на Правен факултет?",
    )

    assert result.outcome == expected_outcome
    assert result.constraints is None


def test_planner_propagates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(constraints_module, "get_question_names_query", cancelled)
    with pytest.raises(asyncio.CancelledError):
        anyio.run(
            plan_faq_evidence_constraints,
            Database("postgresql://unused"),
            "Каде е службата на Правен факултет?",
        )


@pytest.mark.parametrize("mode", list(QueryTransformMode))
@pytest.mark.parametrize("exclusion_first", [False, True])
@pytest.mark.parametrize("institution", ["Правен факултет", "ФИНКИ"])
@pytest.mark.parametrize("office", ["Деканатот", "Архивата"])
def test_compound_scope_and_exclusion_are_both_enforced_before_rerank(
    monkeypatch: pytest.MonkeyPatch,
    mode: QueryTransformMode,
    exclusion_first: bool,
    institution: str,
    office: str,
) -> None:
    request = f"Каде е {office} на {institution}?"
    exclusion = "Не прашувам за студентската служба."
    query = f"{exclusion} {request}" if exclusion_first else f"{request} {exclusion}"
    names = AsyncMock(return_value=list(_OFFICES))
    monkeypatch.setattr(constraints_module, "get_question_names_query", names)
    candidates = [
        _faq("Студентска служба", content="070 302 440, Студентска служба."),
        _faq("Деканат"),
        _faq("Архива", content="Архивата е во Деканатот."),
        _faq("Контакт и локација", content="070 302 440, Студентска служба."),
        _faq("Уписи"),
    ]
    chunk = ChunkSchema(
        id=uuid4(),
        document_id=uuid4(),
        document_name="procedures",
        document_title="Документ за постапки",
        chunk_index=0,
        content="Документ за постапки за упис.",
        distance=0.2,
    )
    payloads: list[dict[str, object]] = []

    class Response:
        def json(self):
            documents = payloads[-1]["documents"]
            assert isinstance(documents, list)
            return {
                "reranked_documents": [
                    {"index": index, "score": 0.95} for index in range(len(documents))
                ],
            }

    async def rerank(payload):
        payloads.append(payload)
        return Response()

    planned_queries = _install_pipeline_mocks(
        monkeypatch, mode=mode, candidates=candidates, rerank=rerank
    )
    monkeypatch.setattr(
        context_module, "_search_both", AsyncMock(return_value=(candidates, [chunk]))
    )
    monkeypatch.setattr(context_module, "get_chunks_window", AsyncMock(return_value=[]))
    # Rewrites and history may remove both clauses; only the latest query sets policy.
    monkeypatch.setattr(
        context_module,
        "_contextualize_query",
        AsyncMock(return_value="Каде е службата?"),
    )
    exact = AsyncMock(return_value=candidates[0])
    monkeypatch.setattr(recovery_plan_module, "get_question_by_name_query", exact)

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=mode,
            history_text="previous office conversation",
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)

    expected_faqs = (
        ["Деканат", "Архива", "Уписи"] if institution == "ФИНКИ" else ["Уписи"]
    )
    assert planned_queries == [query]
    assert len(payloads) == 1
    documents = payloads[0]["documents"]
    assert isinstance(documents, list)
    assert [text.split("\n")[0] for text in documents] == [
        *(f"Наслов: {title}" for title in expected_faqs),
        "Наслов: Документ за постапки",
    ]
    assert [source.title for source in result.sources] == [
        *expected_faqs,
        chunk.document_title,
    ]
    assert "070 302 440" not in result.text
    assert chunk.content in result.text
    assert timings.faq_evidence_constraint_reason == "entity_exclusion"
    assert timings.faq_evidence_excluded_candidate_count == (
        2 if institution == "ФИНКИ" else 4
    )
    if mode == QueryTransformMode.RAW:
        exact.assert_awaited_once()


@pytest.mark.parametrize(
    ("canonical", "stored_title"),
    [
        ("Студентска служба", "Студентска служба"),
        ("Студентска служба", "студентска служба"),
        ("Студентска служба", "Studentska sluzba"),
        ("Студентска служба", "Studentska sluzhba"),
        ("Деканат", "Деканат"),
        ("Деканат", "деканат"),
        ("Деканат", "Dekanat"),
    ],
)
def test_normalized_exclusion_withholds_actual_catalog_fact_carriers(
    monkeypatch: pytest.MonkeyPatch, canonical: str, stored_title: str
) -> None:
    carrier_title = "kontakt i lokacija"
    catalog = [
        stored_title
        if title == canonical
        else carrier_title
        if title == "Контакт и локација"
        else title
        for title in _OFFICES
    ]
    query = f"Каде е архивата? Не прашувам за {canonical}."
    constraints = parse_explicit_faq_constraints(query, catalog)
    assert constraints is not None
    assert constraints.excluded_title == stored_title
    assert constraints.direct_fact_source_titles == (carrier_title,)
    monkeypatch.setattr(
        constraints_module, "get_question_names_query", AsyncMock(return_value=catalog)
    )
    candidates = [
        _faq(stored_title),
        _faq(
            carrier_title, content="070 302 440, Студентска служба; 02 370 377, Декан."
        ),
        _faq("Архива", content="Архивата е во Деканатот."),
    ]
    payloads: list[dict[str, object]] = []

    class Response:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        payloads.append(payload)
        return Response()

    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=candidates,
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)
    assert len(payloads) == 1
    assert payloads[0]["documents"] == [
        f"Наслов: Архива\nСодржина: {candidates[2].content}"
    ]
    assert [source.title for source in result.sources] == ["Архива"]
    assert "070 302 440" not in result.text
    assert "02 370 377" not in result.text
    assert candidates[2].content in result.text
    assert timings.faq_evidence_excluded_candidate_count == 2


@pytest.mark.parametrize(
    ("query", "title"),
    [
        ("Каде е Деканатот на мапата?", "Деканат"),
        (
            "Кое е работното време на студентската служба на празници?",
            "Студентска служба",
        ),
        ("Каде е студентската служба на втори кат?", "Студентска служба"),
        ("Кога работи Архива на крајот на семестарот?", "Архива"),
        ("Кој е телефонот на Деканатот на професорот од Правен факултет?", "Деканат"),
    ],
)
def test_noninstitution_complements_preserve_ordinary_pipeline(
    monkeypatch: pytest.MonkeyPatch, query: str, title: str
) -> None:
    assert parse_explicit_faq_constraints(query) is None
    assert parse_explicit_faq_constraints(query, _OFFICES) is None
    names = AsyncMock(
        side_effect=AssertionError("ordinary query must not load scope catalog")
    )
    monkeypatch.setattr(constraints_module, "get_question_names_query", names)
    faq = _faq(title)
    payloads: list[dict[str, object]] = []

    class Response:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        payloads.append(payload)
        return Response()

    _install_pipeline_mocks(
        monkeypatch, mode=QueryTransformMode.REWRITE, candidates=[faq], rerank=rerank
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)
    names.assert_not_awaited()
    assert len(payloads) == 1
    assert payloads[0]["documents"] == [f"Наслов: {title}\nСодржина: {faq.content}"]
    assert [source.title for source in result.sources] == [title]
    assert faq.content in result.text
    assert timings.faq_evidence_constraint_applied is False


@pytest.mark.parametrize("exclusion_first", [False, True])
@pytest.mark.parametrize(
    "catalog",
    [
        [title for title in _OFFICES if title != "Деканат"],
        [*_OFFICES, "studentska sluzba"],
        [*_OFFICES, "Kontakt i lokacija"],
        [*_OFFICES, "Kompjuterski centar"],
        [f"FAQ {index}" for index in range(513)],
    ],
)
def test_compound_live_catalog_cannot_drop_either_recognized_constraint(
    monkeypatch: pytest.MonkeyPatch, exclusion_first: bool, catalog: list[str]
) -> None:
    request = "Каде е Деканатот на Правен факултет?"
    exclusion = "Не прашувам за студентската служба."
    query = f"{exclusion} {request}" if exclusion_first else f"{request} {exclusion}"
    preliminary = parse_explicit_faq_constraints(query)
    assert preliminary is not None
    assert preliminary.kind == "exclusion"
    assert preliminary.institution_is_finki is False
    assert preliminary.target_title == "Деканат"
    assert preliminary.excluded_title == "Студентска служба"
    monkeypatch.setattr(
        constraints_module, "get_question_names_query", AsyncMock(return_value=catalog)
    )
    rerank = AsyncMock(side_effect=AssertionError("invalid catalog must fail closed"))
    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=[_faq(title) for title in _OFFICES],
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)
    assert result.text == ""
    assert result.sources == ()
    assert timings.faq_evidence_constraint_reason == "catalog_error"
    assert timings.faq_evidence_constrained_abstained is True
    rerank.assert_not_awaited()


def test_plain_exclusion_does_not_depend_on_unrelated_real_catalog_collision() -> None:
    constraints = parse_explicit_faq_constraints(
        "Каде е архивата? Не прашувам за студентската служба.",
        [*_OFFICES, "Kompjuterski centar"],
    )
    assert constraints is not None
    assert constraints.kind == "exclusion"
    assert constraints.institution_is_finki is None
    assert constraints.excluded_title == "Студентска служба"
    assert constraints.direct_fact_source_titles == ("Контакт и локација",)


@pytest.mark.parametrize(
    ("query", "target", "is_finki"),
    [
        (
            "Каде е Деканатот на Правен факултет? Не прашувам за „студентската служба“.",
            "Деканат",
            False,
        ),
        (
            'Не прашувам за "студентската служба". Каде е Деканатот на ФИНКИ?',
            "Деканат",
            True,
        ),
        (
            "Која е контакт е-поштата на Медицински факултет? Не прашувам за студентската служба.",
            "Контакт и локација",
            False,
        ),
        (
            "Не прашувам за студентската служба. Која е контакт е-поштата на ФИНКИ?",
            "Контакт и локација",
            True,
        ),
    ],
)
def test_compound_parser_reuses_shared_institution_forms_and_quote_rules(
    query: str, target: str, is_finki: bool
) -> None:
    constraints = parse_explicit_faq_constraints(query, _OFFICES)
    assert constraints is not None
    assert constraints.kind == "exclusion"
    assert constraints.institution_is_finki is is_finki
    assert constraints.target_title == target
    assert constraints.excluded_title == "Студентска служба"


@pytest.mark.parametrize(
    "query",
    [
        '"Не прашувам за студентската служба." Каде е Деканатот на Правен факултет?',
        "„Не прашувам за студентската служба.“ Каде е Деканатот на Правен факултет?",
        "Каде е Деканатот на Оксфорд?",
    ],
)
def test_unreviewed_quoted_markers_and_unmarked_institution_names_are_ordinary(
    query: str,
) -> None:
    assert parse_explicit_faq_constraints(query, _OFFICES) is None


@pytest.mark.parametrize("institution", ["ФИНКИ", "Правен факултет"])
@pytest.mark.parametrize(
    "catalog_change",
    [
        None,
        "Kompjuterski centar",
        "Dekanat",
        "Studentska sluzba",
        "Kontakt i lokacija",
        "missing_target",
    ],
)
def test_compound_catalog_collisions_only_block_affected_evidence(
    monkeypatch: pytest.MonkeyPatch,
    institution: str,
    catalog_change: str | None,
) -> None:
    query = f"Каде е деканатот на {institution}? Не прашувам за студентската служба."
    catalog = list(_OFFICES)
    if catalog_change == "missing_target":
        catalog.remove("Деканат")
    elif catalog_change is not None:
        catalog.append(catalog_change)
    monkeypatch.setattr(
        constraints_module, "get_question_names_query", AsyncMock(return_value=catalog)
    )
    candidates = [
        _faq("Студентска служба", content="070 302 440, Студентска служба."),
        _faq("Деканат"),
        _faq("Контакт и локација", content="070 302 440, Студентска служба."),
    ]
    payloads: list[dict[str, object]] = []

    class Response:
        def json(self):
            return {"reranked_documents": [{"index": 0, "score": 0.95}]}

    async def rerank(payload):
        payloads.append(payload)
        return Response()

    _install_pipeline_mocks(
        monkeypatch,
        mode=QueryTransformMode.REWRITE,
        candidates=candidates,
        rerank=rerank,
    )

    async def run():
        return await context_module.get_retrieved_context_with_sources(
            Database("postgresql://unused"),
            query,
            Model.BGE_M3_LOCAL,
            Model.GPT_5_4_MINI,
            query_transform_mode=QueryTransformMode.REWRITE,
        )

    timings, token = start_request_timings()
    try:
        result = anyio.run(run)
    finally:
        reset_request_timings(token)
    catalog_error = catalog_change not in {None, "Kompjuterski centar"} or (
        institution != "ФИНКИ" and catalog_change == "Kompjuterski centar"
    )
    if catalog_error:
        assert result.text == ""
        assert result.sources == ()
        assert payloads == []
        assert timings.faq_evidence_constraint_reason == "catalog_error"
        assert timings.faq_evidence_constrained_abstained is True
    elif institution == "ФИНКИ":
        assert [source.title for source in result.sources] == ["Деканат"]
        assert candidates[1].content in result.text
        assert "070 302 440" not in result.text
        assert len(payloads) == 1
        assert payloads[0]["documents"] == [
            f"Наслов: Деканат\nСодржина: {candidates[1].content}"
        ]
        assert timings.faq_evidence_constraint_reason == "entity_exclusion"
        assert timings.faq_evidence_excluded_candidate_count == 2
    else:
        assert result.text == ""
        assert result.sources == ()
        assert payloads == []
        assert timings.faq_evidence_constraint_reason == "entity_exclusion"
        assert timings.faq_evidence_excluded_candidate_count == 3
