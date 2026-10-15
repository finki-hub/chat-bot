import pytest

from app.llms.query_normalization import (
    query_search_variants,
    romanized_macedonian_variant,
)


def test_cyrillic_query_keeps_only_original_variant() -> None:
    query = "Кои смерови ги нуди ФИНКИ?"

    variants = query_search_variants(query)

    assert variants == (query,)


def test_romanized_macedonian_adds_cyrillic_variant() -> None:
    query = "sakam da se zapisham na FINKI"

    variants = query_search_variants(query)

    assert variants == (query, "сакам да се запишам на FINKI")


def test_transliteration_handles_macedonian_digraphs() -> None:
    query = "gj kj zh dz dzh lj nj ch sh"

    variants = query_search_variants(query)

    assert variants == (query, "ѓ ќ ж ѕ џ љ њ ч ш")


def test_mixed_identifiers_are_preserved() -> None:
    query = "Shto e SEIS23 i kade e iKnow?"

    variants = query_search_variants(query)

    assert variants == (query, "Што е SEIS23 и каде е iKnow?")


def test_non_alphabetic_query_does_not_add_duplicate_variant() -> None:
    query = "12345?!"

    variants = query_search_variants(query)

    assert variants == (query,)


def test_standard_macedonian_diacritics_are_transliterated() -> None:
    query = "što e član so žeton ǵevrek i ḱebe"

    variants = query_search_variants(query)

    assert variants == (query, "што е члан со жетон ѓеврек и ќебе")


def test_unsupported_unicode_token_is_preserved_whole() -> None:
    query = "café smerovi"

    variants = query_search_variants(query)

    assert variants == (query, "café смерови")


def test_decomposed_macedonian_diacritics_are_normalized_before_transliteration() -> (
    None
):
    query = "s\u030cto g\u0301evrek k\u0301ebe"

    variants = query_search_variants(query)

    assert variants == (query, "што ѓеврек ќебе")


def test_decomposed_unsupported_token_is_preserved_whole() -> None:
    query = "cafe\u0301 smerovi"

    variants = query_search_variants(query)

    assert variants == (query, "café смерови")


def test_normalization_only_change_does_not_add_search_variant() -> None:
    query = "cafe\u0301"

    variants = query_search_variants(query)

    assert variants == (query,)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Kade e sluzba", "Каде е слузба"),
        ("Kade e služba", "Каде е служба"),
        ("koga se zimskite kolokviumi", "кога се зимските колоквиуми"),
        ("sho se uchi po MIS?", "шо се учи по MIS?"),
        ("članarina", "чланарина"),
    ],
)
def test_romanized_macedonian_variant_transliterates_macedonian(
    query: str,
    expected: str,
) -> None:
    assert romanized_macedonian_variant(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Web programming",
        "machine learning exam schedule",
        "Is e-commerce taught at FINKI?",
        "Каде е службата?",
        "Kade e службата?",
        "sluzba",
    ],
)
def test_romanized_macedonian_variant_skips_english_cyrillic_and_unmarked(
    query: str,
) -> None:
    assert romanized_macedonian_variant(query) is None
