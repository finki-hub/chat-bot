from dataclasses import dataclass
from typing import Final

_MIN_INITIAL_K: Final[int] = 30


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    initial_k: int
    per_query_k: int


def retrieval_budget(
    requested_initial_k: int,
    *,
    variant_count: int = 1,
) -> RetrievalBudget:
    """Split the candidate budget across the query variants."""
    initial_k = max(requested_initial_k, _MIN_INITIAL_K)
    return RetrievalBudget(
        initial_k=initial_k,
        per_query_k=initial_k // variant_count + 1,
    )
