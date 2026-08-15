from collections.abc import Iterable


def lexical_faq_expansion_allowed(
    faq_distances: Iterable[float | None],
    chunk_distances: Iterable[float | None],
    *,
    has_transliterated_query: bool = False,
) -> bool:
    """Return whether dense retrieval identifies FAQ as the nearest source type."""
    scored_faqs = [distance for distance in faq_distances if distance is not None]
    scored_chunks = [distance for distance in chunk_distances if distance is not None]
    if has_transliterated_query:
        return bool(scored_faqs or scored_chunks)
    if not scored_faqs:
        return False
    return not scored_chunks or min(scored_faqs) <= min(scored_chunks)
