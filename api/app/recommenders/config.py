from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    # --- defense signal (original) ---
    similarity_weight: float = 1.0
    rerank_weight: float = 1.0
    recency_half_life_days: float = 0.0  # 0 = off
    pair_affinity_weight: float = (
        0.5  # two strong individuals vs a pair that has actually served together
    )
    # Mentor-conditioned habitual co-membership prior (title-independent), built from the
    # WHOLE defense graph, applied once the mentor is resolved. Committee composition is
    # habitual, so "who this mentor usually sits with" surfaces the right members even when
    # the topical retrieval missed them (the cold-start majority). 0.0 reproduces the
    # retrieval-only member scoring exactly (ablation pivot). Swept on the GATE A corpus:
    # MEMBERS-ONLY member-pair Jaccard 0.135 (w=0) -> 0.292 (w=3), recovering the strong
    # habitual-co-membership baseline that retrieval-only scoring was missing. Plateaus by
    # w~3-6; 3.0 is the lowest weight at the peak, leaving the most room for the topical
    # (and future paper/buddy) signals to re-rank within the mentor's collaborators.
    mentor_prior_weight: float = 3.0
    mentor_topk: int = 3
    # --- expertise signal (papers) ---
    # Per-professor topical expertise from the paper corpus, a GENTLE additive lift on member
    # candidates. Swept on GATE B: ~0.5 is neutral-to-slightly-positive on historical-match
    # metrics (members 0.301->0.302, FULL pair 0.121->0.123 on the full population) — papers
    # barely move history-replication because committee composition is habitual (the mentor
    # prior dominates), but expertise adds a real topical-qualification signal + an
    # explainability trail. Weights >~1 OVERRIDE the prior with topical-but-non-habitual
    # people and HURT. 0.0 reproduces the defenses-only result exactly (ablation pivot).
    expertise_weight: float = 0.5
    expertise_top_papers: int = 5  # top-N similarities summed per professor
    expertise_recency_half_life_days: float = 0.0
    rerank_papers: bool = False
    # --- co-author / buddy signal (papers) ---
    coauthor_weight: float = (
        0.0  # 0 reproduces the defenses+expertise result EXACTLY (ablation pivot)
    )
    coauthor_recency_half_life_days: float = (
        0.0  # 0 = recency off; in days for kernel reuse, applied to year deltas (x365)
    )
    coauthor_member_boost: float = 0.0  # MEMBERS-ONLY: boost candidates who are topic-recent buddies of the GIVEN mentor
    # GLOBAL co-author prior: the resolved mentor's frequent co-authors over the WHOLE paper
    # graph (title-independent), applied like the mentor co-membership prior. Co-authorship
    # predicts committee co-membership (~2.9x lift in the corpus); its UNIQUE value over the
    # defense prior is surfacing co-authors who have not YET served together (cold-start).
    coauthor_prior_weight: float = 0.0
    # --- cold-start lever ---
    cold_start_defense_floor: int = (
        2  # professors with < this many retrieved defenses ...
    )
    cold_start_expertise_boost: float = 1.5  # ... get expertise_weight * this
    # --- members-only mode ---
    mentor_match_boost: float = (
        0.0  # up-weight retrieved defenses where the given mentor served
    )
