from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    similarity_weight: float = 1.0
    rerank_weight: float = 1.0
    recency_half_life_days: float = 1825.0
    pair_affinity_weight: float = 0.5
    mentor_prior_weight: float = 1.5
    mentor_topk: int = 3
    expertise_weight: float = (
        0.5  # swept; >~1 overrides the prior with non-habitual people
    )
    expertise_top_papers: int = 5
    expertise_recency_half_life_days: float = 1825.0
    rerank_papers: bool = False
    coauthor_weight: float = 0.0  # off; co-authorship didn't help member-pair metrics
    coauthor_recency_half_life_days: float = 0.0
    coauthor_member_boost: float = 0.0
    coauthor_prior_weight: float = 0.0
    cold_start_defense_floor: int = 2
    cold_start_expertise_boost: float = 1.5
    mentor_match_boost: float = 0.0
