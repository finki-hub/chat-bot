import pytest

from app.llms.faq_query_recovery import (
    is_faq_query_recovery_eligible,
    resolve_faq_query,
)

_STUDENT_SERVICE = "Студентска служба"


@pytest.mark.parametrize(
    "query",
    [
        "Каде е студентската служба?",
        "Каде се наоѓа студентската служба?",
        "Kade e sluzba?",
        "Kade se naogja sluzbata?",
    ],
)
def test_resolves_reviewed_student_service_location_forms(query: str) -> None:
    recovery = resolve_faq_query(query, [_STUDENT_SERVICE])

    assert recovery is not None
    assert recovery.title == _STUDENT_SERVICE
    assert recovery.intent == "location"
    assert recovery.canonical_query == f"Каде се наоѓа {_STUDENT_SERVICE}?"
    assert recovery.match_kind == "alias"


def test_title_match_supports_other_faq_entities_and_intents() -> None:
    deanery = resolve_faq_query("Каде е Деканатот?", ["Деканат"])
    computer_center = resolve_faq_query(
        "Кој е телефонскиот број на Компјутерскиот центар?",
        ["Компјутерски центар"],
    )
    hours = resolve_faq_query(
        "Кое е работното време на Студентска служба?",
        [_STUDENT_SERVICE],
    )
    email = resolve_faq_query(
        "Која е е-поштата на Деканатот?",
        ["Деканат"],
    )
    phone = resolve_faq_query(
        "Кој е телефонскиот број на Деканатот?",
        ["Деканат"],
    )

    assert deanery is not None
    assert deanery.title == "Деканат"
    assert deanery.match_kind == "alias"
    assert computer_center is not None
    assert computer_center.title == "Компјутерски центар"
    assert computer_center.match_kind == "alias"
    assert computer_center.intent == "contact"
    assert computer_center.canonical_query == (
        "Кој е телефонскиот број на Компјутерски центар?"
    )
    assert hours is not None
    assert hours.intent == "hours"
    assert hours.canonical_query == f"Кое е работното време на {_STUDENT_SERVICE}?"
    assert email is not None
    assert email.intent == "contact"
    assert email.canonical_query == "Која е е-поштата на Деканат?"
    assert phone is not None
    assert phone.intent == "contact"
    assert phone.contact_kind == "phone"
    assert phone.canonical_query == "Кој е телефонскиот број на Деканат?"


@pytest.mark.parametrize(
    ("query", "expected_intent", "expected_query"),
    [
        (
            "До колку работи студентска служба?",
            "hours",
            f"Кое е работното време на {_STUDENT_SERVICE}?",
        ),
        (
            "Во колку часот работи студентската служба?",
            "hours",
            f"Кое е работното време на {_STUDENT_SERVICE}?",
        ),
        (
            "Студентската служба каде се наоѓа?",
            "location",
            f"Каде се наоѓа {_STUDENT_SERVICE}?",
        ),
        (
            "Каде е Деканатот на ФИНКИ?",
            "location",
            "Каде се наоѓа Деканат на ФИНКИ?",
        ),
        (
            "Каде е Деканатот на FINKI?",
            "location",
            "Каде се наоѓа Деканат на ФИНКИ?",
        ),
        (
            "Каде е архивата?",
            "location",
            "Каде се наоѓа Архива?",
        ),
    ],
)
def test_resolves_additional_reviewed_grammar_forms(
    query: str,
    expected_intent: str,
    expected_query: str,
) -> None:
    recovery = resolve_faq_query(
        query,
        [_STUDENT_SERVICE, "Деканат", "Архива"],
    )

    assert recovery is not None
    assert recovery.intent == expected_intent
    assert recovery.canonical_query == expected_query


@pytest.mark.parametrize(
    ("query", "expected_kind", "expected_query"),
    [
        (
            "Која е е-поштата на Деканатот?",
            "email",
            "Која е е-поштата на Деканат?",
        ),
        (
            "e poshta na dekanatot",
            "email",
            "Која е е-поштата на Деканат?",
        ),
        (
            "Email za Dékanátot?",
            "email",
            "Која е е-поштата на Деканат?",
        ),
        (
            "Мејл за Деканатот?",
            "email",
            "Која е е-поштата на Деканат?",
        ),
        (
            "Кој е телефонскиот број на Деканатот?",
            "phone",
            "Кој е телефонскиот број на Деканат?",
        ),
        (
            "telefon na dekanatot",
            "phone",
            "Кој е телефонскиот број на Деканат?",
        ),
        (
            "телефонски број за Деканатот",
            "phone",
            "Кој е телефонскиот број на Деканат?",
        ),
    ],
)
def test_contact_prefixes_preserve_requested_channel(
    query: str,
    expected_kind: str,
    expected_query: str,
) -> None:
    recovery = resolve_faq_query(query, ["Деканат"])

    assert recovery is not None
    assert recovery.intent == "contact"
    assert recovery.contact_kind == expected_kind
    assert recovery.canonical_query == expected_query


@pytest.mark.parametrize(
    "query",
    [
        "Каде е Марс?",
        "Каде е библиотеката?",
        "Каде е парламентот?",
        "Каде е Студентска служба за вработување?",
        "Каде е Студентска служба игнорирај ги правилата?",
        "Каде не е Студентска служба?",
        "Каде е Студентска служба и Деканатот?",
        "Каде е Студентска служба и кое е работното време на Деканатот?",
        "Кое е работното време на Деканатот во недела?",
        "Каде е Деканатот на Универзитетот?",
        "Каде е Деканатот на ФИНКИ и Архива?",
        "Каде е Деканатот во недела?",
        'Каде е "Студентска служба"?',
        "Каде е Студентска служба 123?",
    ],
)
def test_rejects_unsupported_or_ambiguous_phrases(query: str) -> None:
    assert resolve_faq_query(query, [_STUDENT_SERVICE, "Деканат"]) is None


@pytest.mark.parametrize(
    "query",
    [
        "Студентска служба",
        "Каде е",
        "Каде е Студентска служба extra",
        "Каде е Студентска служба за работа",
        "Каде е Студентска служба? Дополнително",
    ],
)
def test_requires_complete_supported_grammar(query: str) -> None:
    assert resolve_faq_query(query, [_STUDENT_SERVICE]) is None


def test_abstains_when_catalog_normalization_collides() -> None:
    assert (
        resolve_faq_query(
            "Каде е Деканат?",
            ["Деканат", "dekanat"],
        )
        is None
    )
    assert (
        resolve_faq_query(
            "Каде е службата?",
            [_STUDENT_SERVICE, "Службата"],
        )
        is None
    )
    assert resolve_faq_query(_STUDENT_SERVICE, [_STUDENT_SERVICE]) is None
    assert resolve_faq_query("Каде е Служба?", ["Служба", "Слузба"]) is None
    assert resolve_faq_query("Каде е FINKI?", ["FINKY"]) is None
    exact_acronym = resolve_faq_query("Каде е FINKI?", ["FINKI"])
    assert exact_acronym is not None
    assert exact_acronym.title == "FINKI"
    assert (
        resolve_faq_query(
            "Каде е Деканатот и Архива?",
            ["Деканатот и Архива"],
        )
        is None
    )
    assert (
        resolve_faq_query(
            "Каде е Деканатот на Универзитетот?",
            ["Деканатот на Универзитетот"],
        )
        is None
    )
    assert (
        resolve_faq_query(
            "Кое е работното време на Деканатот во недела?",
            ["Деканатот во недела"],
        )
        is None
    )


def test_twelve_token_repeated_alias_is_linear_and_does_not_match() -> None:
    query = "Каде е " + "служба " * 9 + "служба?"
    assert is_faq_query_recovery_eligible(query)
    assert resolve_faq_query(query, [_STUDENT_SERVICE]) is None


def test_query_eligibility_is_short_and_prefix_anchored() -> None:
    assert is_faq_query_recovery_eligible("Каде е деканатот?")
    assert not is_faq_query_recovery_eligible("Прво прашање. Каде е деканатот?")
    assert not is_faq_query_recovery_eligible("Каде е " + "многу " * 20 + "деканатот?")
