from dataclasses import dataclass
from typing import Final

from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import query_variant_count

_MIN_INITIAL_K_BY_MODE: Final[dict[QueryTransformMode, int]] = {
    QueryTransformMode.RAW: 30,
    QueryTransformMode.REWRITE: 40,
    QueryTransformMode.HYDE: 40,
    QueryTransformMode.REWRITE_HYDE: 60,
}


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    initial_k: int
    per_query_k: int


def retrieval_budget(
    mode: QueryTransformMode,
    requested_initial_k: int,
) -> RetrievalBudget:
    initial_k = max(requested_initial_k, _MIN_INITIAL_K_BY_MODE[mode])
    return RetrievalBudget(
        initial_k=initial_k,
        per_query_k=initial_k // query_variant_count(mode) + 1,
    )
