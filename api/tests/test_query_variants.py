from app.llms import query_variants


def test_romanized_macedonian_query_adds_cyrillic_variant() -> None:
    result = query_variants.build_query_variants("Kade e sluzba")

    assert tuple(variant.kind for variant in result.variants) == (
        "transliteration",
        "raw",
    )
    assert result.variants[0] == query_variants.QueryVariant(
        kind="transliteration",
        text="Каде е слузба",
        is_document=False,
    )
    assert len(result.variants) == 2
    assert result.rerank_query == "Каде е слузба"


def test_cyrillic_query_is_searched_raw_only() -> None:
    search_query = "Како се пријавува испит?"

    result = query_variants.build_query_variants(search_query)

    assert result.variants == (
        query_variants.QueryVariant(
            kind="raw",
            text=search_query,
            is_document=False,
        ),
    )
    assert result.rerank_query == search_query


def test_english_query_adds_no_transliteration_variant() -> None:
    result = query_variants.build_query_variants("Web programming")

    assert tuple(variant.kind for variant in result.variants) == ("raw",)
    assert len(result.variants) == 1
    assert result.rerank_query == "Web programming"
