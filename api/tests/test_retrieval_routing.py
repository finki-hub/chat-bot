from app.llms.retrieval_routing import lexical_faq_expansion_allowed


def test_lexical_faq_expansion_requires_dense_faq_signal() -> None:
    assert lexical_faq_expansion_allowed([], []) is False
    assert lexical_faq_expansion_allowed([], [0.2]) is False


def test_lexical_faq_expansion_accepts_faq_only_signal() -> None:
    assert lexical_faq_expansion_allowed([0.3], []) is True


def test_lexical_faq_expansion_accepts_closer_faq_signal() -> None:
    assert lexical_faq_expansion_allowed([0.2, 0.4], [0.3]) is True


def test_lexical_faq_expansion_rejects_closer_document_signal() -> None:
    assert lexical_faq_expansion_allowed([0.4], [0.2, 0.3]) is False


def test_lexical_faq_expansion_accepts_document_signal_for_transliterated_query() -> (
    None
):
    assert (
        lexical_faq_expansion_allowed(
            [],
            [0.2],
            has_transliterated_query=True,
        )
        is True
    )


def test_lexical_faq_expansion_still_requires_domain_signal_for_transliterated_query() -> (
    None
):
    assert (
        lexical_faq_expansion_allowed(
            [],
            [],
            has_transliterated_query=True,
        )
        is False
    )


def test_lexical_faq_expansion_ignores_unscored_entries() -> None:
    assert lexical_faq_expansion_allowed([None, 0.2], [None, 0.3]) is True
