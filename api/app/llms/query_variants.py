from dataclasses import dataclass
from typing import Literal

from app.llms.query_normalization import romanized_macedonian_variant

QueryVariantKind = Literal["raw", "transliteration"]


@dataclass(frozen=True, slots=True)
class QueryVariant:
    kind: QueryVariantKind
    text: str
    is_document: bool


@dataclass(frozen=True, slots=True)
class QueryVariantBundle:
    variants: tuple[QueryVariant, ...]
    rerank_query: str


def build_query_variants(search_query: str) -> QueryVariantBundle:
    raw = QueryVariant(kind="raw", text=search_query, is_document=False)
    transliterated = romanized_macedonian_variant(search_query)
    if transliterated is None:
        return QueryVariantBundle(variants=(raw,), rerank_query=search_query)
    # Romanized queries embed far from the Cyrillic corpus, so the Cyrillic form is
    # searched too and the reranker compares candidates against it.
    return QueryVariantBundle(
        variants=(
            QueryVariant(
                kind="transliteration", text=transliterated, is_document=False
            ),
            raw,
        ),
        rerank_query=transliterated,
    )
