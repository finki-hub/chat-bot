"""Committee-recommender backtest — GATE A (the quality gate).

Pure-read, offline. The real proof the feature works, run BEFORE the recommend endpoint
is exposed in chat. It imports the production pure functions and the production retrieval
orchestrator — it never reimplements scoring or retrieval, or it would measure the wrong
thing.

Run it inside the api project (it imports the app package and needs DATABASE_URL +
gpu-api/reranker access), with cwd = api so `resources/` and Settings resolve:

    cd api && uv run python ../scripts/backtest.py --sample-size 300
    cd api && uv run python ../scripts/backtest.py --full --mode both

Method — leave-one-out, single retrieval, multi-eval:
  Population = defended diplomas with a non-null embedding + present mentor + BOTH members
  (the only rows with fully-defined ground truth). For each held-out diploma we call
  `retrieve_similar_diplomas(db, held.title, model, initial_k, top_k,
  exclude_external_id=held.external_id)` ONCE — this re-embeds the held-out TITLE ONLY
  (E5 `query:`; production realism: the input shape users actually send) and excludes the
  held-out defense row (the leave-one-out leakage guard). From that single retrieved set
  we evaluate BOTH modes:

    FULL          -> score_people(..., Mode.FULL) -> select_committee(..., Mode.FULL)
                     metrics: mentor hit@1, mentor hit@3, member-pair Jaccard
    MEMBERS_ONLY  -> score_people(..., Mode.MEMBERS_ONLY, given_mentor=true_mentor)
                     -> select_committee(..., given_mentor=true_mentor)
                     metric: member-pair Jaccard only (mentor is GIVEN, not predicted)

Baselines (reported alongside the model):
  FULL mentor          -> globally-most-frequent mentor in the population
  MEMBERS_ONLY pair    -> most-frequent co-members of the given mentor (defense graph)

The GATE: the model must beat the most-frequent-mentor baseline (FULL mentor hit@1/@3),
and both modes' member-pair Jaccard should beat their respective pair baselines.

GATE A is defenses-only by construction: `expertise_weight` and `coauthor_weight` default
to 0, so `score_people` is numerically identical to the original defense-only scorer until
a paper/buddy knob is swept. `--no-papers`/`--no-buddies` force those to 0 explicitly
(forward-compatible no-ops at GATE A; meaningful once the paper corpus lands at GATE B).
"""

# ruff: noqa: INP001 - standalone offline script; scripts/ is not an importable package

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

# The `app` package is not installed; it is the `api/app` source tree resolved via cwd.
# Run from `api` (`cd api && uv run python ../scripts/backtest.py`): Python seeds
# sys.path[0] with this script's dir (scripts/), not the cwd, so add cwd if `app` is there
# and not already importable. This mirrors `PYTHONPATH=/app` in the container invocation.
if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from asyncpg import Record

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.data.connection import Database
from app.data.diplomas import get_backtest_population
from app.data.professor_documents import get_all_paper_authors
from app.llms.diploma_retrieval import retrieve_similar_diplomas
from app.llms.models import Model
from app.llms.professor_retrieval import retrieve_professor_papers
from app.recommenders.config import ScoringWeights
from app.recommenders.recommend import (
    CoauthorIndex,
    ExpertiseIndex,
    MentorPriorIndex,
    Mode,
    RetrievedDiploma,
    _accumulate_coauthor_edges,
    build_coauthor_prior,
    build_expertise_index,
    build_mentor_prior,
    score_people,
    select_committee,
)
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

settings = Settings()

# Empty paper-derived indexes: GATE A is defenses-only, so there is no expertise/buddy
# signal. With expertise_weight=coauthor_weight=0 these are inert anyway, but they make
# the "no paper corpus yet" reality explicit and keep the score_people call signature
# identical to production.
EMPTY_EXPERTISE = ExpertiseIndex(by_professor={}, supporting={})
EMPTY_COAUTHORS = CoauthorIndex(edges={}, supporting={})


@dataclass(frozen=True)
class HeldOut:
    external_id: str
    title: str
    mentor: str
    member1: str
    member2: str

    @property
    def true_pair(self) -> frozenset[str]:
        return frozenset({self.member1, self.member2})


def _jaccard(predicted: Sequence[str], truth: frozenset[str]) -> float:
    pred = frozenset(predicted)
    union = pred | truth
    if not union:
        return 0.0
    return len(pred & truth) / len(union)


def _mentor_topk_names(ranked_mentor_score: dict[str, float], k: int) -> list[str]:
    return [
        name
        for name, _ in sorted(
            ranked_mentor_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
    ]


@dataclass
class CaseResult:
    held: HeldOut
    full_mentor_hit1: bool
    full_mentor_hit3: bool
    full_pair_jaccard: float
    members_pair_jaccard: float
    is_cold_start: bool
    # baselines (computed from the same retrieved set / population)
    base_mentor_hit1: bool
    base_mentor_hit3: bool
    base_full_pair_jaccard: float
    base_members_pair_jaccard: float


def build_population(rows: list[Record]) -> list[HeldOut]:
    return [
        HeldOut(
            external_id=row["external_id"],
            title=row["title"],
            mentor=row["mentor"],
            member1=row["member1"],
            member2=row["member2"],
        )
        for row in rows
    ]


def stratified_sample(
    population: list[HeldOut],
    sample_size: int,
    rng: random.Random,
) -> list[HeldOut]:
    """Stratified-by-mentor down-sample to ~sample_size cases.

    Proportional allocation per mentor (at least one case per mentor that has any), so the
    mentor distribution of the sample mirrors the full population rather than over-weighting
    prolific mentors. Returns the full population unchanged if it already fits.
    """
    if sample_size <= 0 or sample_size >= len(population):
        return list(population)

    by_mentor: dict[str, list[HeldOut]] = defaultdict(list)
    for case in population:
        by_mentor[case.mentor].append(case)

    total = len(population)
    sampled: list[HeldOut] = []
    for cases in by_mentor.values():
        # proportional quota, floored to >=1 so every mentor stratum is represented
        quota = max(1, round(sample_size * len(cases) / total))
        quota = min(quota, len(cases))
        sampled.extend(rng.sample(cases, quota))

    # Flooring/rounding can overshoot; trim back down deterministically.
    rng.shuffle(sampled)
    return sampled[:sample_size]


def weights_from_args(ns: argparse.Namespace) -> ScoringWeights:
    """Build ScoringWeights from CLI overrides on top of the dataclass defaults.

    `--no-papers` forces expertise_weight=0; `--no-buddies` forces coauthor_weight=0 and
    coauthor_member_boost=0 (composes with --no-papers). At GATE A these are no-ops (both
    already default to 0) but keep the script forward-compatible with the paper corpus.
    """
    base = ScoringWeights()

    # Each CLI value or the dataclass default; explicit kwargs keep the field types sound
    # (every override here is a float field). --no-* ablation switches win over --*-weight.
    expertise_weight = (
        0.0 if ns.no_papers else _or(ns.expertise_weight, base.expertise_weight)
    )
    coauthor_weight = (
        0.0 if ns.no_buddies else _or(ns.coauthor_weight, base.coauthor_weight)
    )
    coauthor_member_boost = (
        0.0
        if ns.no_buddies
        else _or(ns.coauthor_member_boost, base.coauthor_member_boost)
    )

    return replace(
        base,
        similarity_weight=_or(ns.similarity_weight, base.similarity_weight),
        rerank_weight=_or(ns.rerank_weight, base.rerank_weight),
        recency_half_life_days=_or(
            ns.recency_half_life_days,
            base.recency_half_life_days,
        ),
        pair_affinity_weight=_or(ns.pair_affinity_weight, base.pair_affinity_weight),
        mentor_prior_weight=_or(ns.mentor_prior_weight, base.mentor_prior_weight),
        coauthor_prior_weight=_or(
            ns.coauthor_prior_weight,
            base.coauthor_prior_weight,
        ),
        expertise_weight=expertise_weight,
        coauthor_weight=coauthor_weight,
        coauthor_member_boost=coauthor_member_boost,
        coauthor_recency_half_life_days=_or(
            ns.coauthor_recency_half_life_days,
            base.coauthor_recency_half_life_days,
        ),
    )


def _or(override: float | None, default: float) -> float:
    return default if override is None else override


def _loo_prior(global_prior: MentorPriorIndex, held: HeldOut) -> MentorPriorIndex:
    """The global mentor prior with the held-out defense's own co-memberships removed.

    Leave-one-out for the prior, mirroring the baseline fix: without it a held-out defense
    leaks its own members into its mentor's prior. Only the held mentor's bucket needs
    adjusting — the held-out defense contributes to no other mentor's counts.
    """
    adjusted = dict(global_prior.by_mentor.get(held.mentor, {}))
    for member in (held.member1, held.member2):
        if member in adjusted:
            adjusted[member] -= 1.0
            if adjusted[member] <= 0.0:
                del adjusted[member]
    return MentorPriorIndex(by_mentor={**global_prior.by_mentor, held.mentor: adjusted})


def evaluate_case(
    held: HeldOut,
    retrieved: list[RetrievedDiploma],
    weights: ScoringWeights,
    *,
    mentor_topk: int,
    run_full: bool,
    run_members: bool,
    cold_start_floor: int,
    global_top_mentor: str,
    mentor_cofreq: dict[str, Counter[str]],
    global_prior: MentorPriorIndex,
    expertise: ExpertiseIndex,
    coauthors: CoauthorIndex,
    coauthor_prior: MentorPriorIndex | None = None,
) -> CaseResult:
    """Evaluate both modes for one held-out case from its single retrieved set.

    Cold-start = the true mentor appears in FEWER than `cold_start_floor` of the retrieved
    defenses (the held-out row is already excluded by retrieval), so the paper/buddy signals
    would have to carry it. Baselines are computed against the same ground truth.
    """
    # cold-start: count the true mentor's OTHER retrieved defenses (any role excludes them
    # being the same row — the held-out is excluded from retrieval).
    mentor_retrieved_defenses = sum(1 for r in retrieved if r.mentor == held.mentor)
    is_cold_start = mentor_retrieved_defenses < cold_start_floor

    loo_prior = _loo_prior(global_prior, held)

    full_hit1 = full_hit3 = False
    full_pair_jaccard = 0.0
    if run_full:
        ranked_full = score_people(
            retrieved,
            expertise,
            coauthors,
            weights,
            Mode.FULL,
        )
        rec_full = select_committee(
            ranked_full,
            Mode.FULL,
            given_mentor=None,
            mentor_topk=mentor_topk,
            mentor_prior=loo_prior,
            coauthor_prior=coauthor_prior,
        )
        full_hit1 = rec_full.mentor == held.mentor
        top3 = _mentor_topk_names(ranked_full.mentor_score, 3)
        full_hit3 = held.mentor in top3
        full_pair_jaccard = _jaccard(rec_full.members, held.true_pair)

    members_pair_jaccard = 0.0
    if run_members:
        ranked_members = score_people(
            retrieved,
            expertise,
            coauthors,
            weights,
            Mode.MEMBERS_ONLY,
            given_mentor=held.mentor,
        )
        rec_members = select_committee(
            ranked_members,
            Mode.MEMBERS_ONLY,
            given_mentor=held.mentor,
            mentor_topk=mentor_topk,
            mentor_prior=loo_prior,
            coauthor_prior=coauthor_prior,
        )
        members_pair_jaccard = _jaccard(rec_members.members, held.true_pair)

    # --- baselines ---
    base_hit1 = global_top_mentor == held.mentor
    base_hit3 = (
        global_top_mentor == held.mentor
    )  # single-name baseline -> same as hit@1
    # FULL pair baseline: the given mentor is NOT known in FULL, so the fairest naive
    # member baseline is the most-frequent co-members of the model's chosen-or-true mentor.
    # We use the held-out's true mentor's top co-members as the naive pair (the strongest
    # naive baseline a FULL system could field if it nailed the mentor).
    #
    # Leave-one-out for the baseline TOO: subtract the held-out defense's own members from
    # its mentor's co-membership counts. Without this, low-frequency mentors leak their true
    # pair into the baseline (a mentor with a single defense would score Jaccard 1.0 for
    # free), making the comparison unfair to the model, whose retrieval already excludes the
    # held-out row.
    mentor_counter = mentor_cofreq.get(held.mentor, Counter()).copy()
    mentor_counter[held.member1] -= 1
    mentor_counter[held.member2] -= 1
    base_pair_names = [
        name for name, count in mentor_counter.most_common() if count > 0
    ][:2]
    base_full_pair_jaccard = _jaccard(tuple(base_pair_names), held.true_pair)
    # MEMBERS_ONLY pair baseline: most-frequent co-members of the GIVEN mentor.
    base_members_pair_jaccard = base_full_pair_jaccard

    return CaseResult(
        held=held,
        full_mentor_hit1=full_hit1,
        full_mentor_hit3=full_hit3,
        full_pair_jaccard=full_pair_jaccard,
        members_pair_jaccard=members_pair_jaccard,
        is_cold_start=is_cold_start,
        base_mentor_hit1=base_hit1,
        base_mentor_hit3=base_hit3,
        base_full_pair_jaccard=base_full_pair_jaccard,
        base_members_pair_jaccard=base_members_pair_jaccard,
    )


def compute_baselines(
    population: list[HeldOut],
) -> tuple[str, dict[str, Counter[str]]]:
    """Naive baselines from the full population (NOT the sample, to be a fair global prior).

    Returns:
      global_top_mentor  -> the single most-frequent mentor (FULL mentor baseline).
      mentor_cofreq      -> mentor -> Counter of co-members (how often each served with that
                            mentor). Raw counts, not pre-sorted, so evaluate_case can do a
                            per-case leave-one-out subtraction before taking the top pair.
    """
    mentor_counts: Counter[str] = Counter(case.mentor for case in population)
    global_top_mentor = mentor_counts.most_common(1)[0][0] if mentor_counts else ""

    co_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for case in population:
        for member in (case.member1, case.member2):
            co_counts[case.mentor][member] += 1
    return global_top_mentor, dict(co_counts)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def report(
    results: list[CaseResult],
    *,
    run_full: bool,
    run_members: bool,
    weights: ScoringWeights,
    model_value: str,
    initial_k: int,
    top_k: int,
    mentor_topk: int,
    population_size: int,
) -> str:
    n = len(results)
    cold = [r for r in results if r.is_cold_start]
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("COMMITTEE-RECOMMENDER BACKTEST — GATE A (defenses-only)")
    lines.append("=" * 72)
    lines.append(
        f"model={model_value}  population={population_size}  evaluated={n}  "
        f"cold_start={len(cold)}",
    )
    lines.append(
        f"initial_k={initial_k}  top_k={top_k}  mentor_topk={mentor_topk}  "
        f"reranker_min={settings.RERANKER_MIN_SCORE}",
    )
    lines.append(
        f"weights: sim={weights.similarity_weight} rerank={weights.rerank_weight} "
        f"recency_hl_days={weights.recency_half_life_days} "
        f"pair_affinity={weights.pair_affinity_weight} "
        f"mentor_prior={weights.mentor_prior_weight} "
        f"coauthor_prior={weights.coauthor_prior_weight} "
        f"expertise={weights.expertise_weight} coauthor={weights.coauthor_weight}",
    )

    def metric_block(title: str, subset: list[CaseResult]) -> None:
        if not subset:
            return
        m = len(subset)
        lines.append("-" * 72)
        lines.append(f"{title} (n={m})")
        if run_full:
            model_h1 = _mean([float(r.full_mentor_hit1) for r in subset])
            base_h1 = _mean([float(r.base_mentor_hit1) for r in subset])
            model_h3 = _mean([float(r.full_mentor_hit3) for r in subset])
            base_h3 = _mean([float(r.base_mentor_hit3) for r in subset])
            model_fp = _mean([r.full_pair_jaccard for r in subset])
            base_fp = _mean([r.base_full_pair_jaccard for r in subset])
            lines.append(
                f"  FULL mentor hit@1     : {model_h1:.3f}  "
                f"(baseline {base_h1:.3f}, {_delta(model_h1, base_h1)})",
            )
            lines.append(
                f"  FULL mentor hit@3     : {model_h3:.3f}  "
                f"(baseline {base_h3:.3f}, {_delta(model_h3, base_h3)})",
            )
            lines.append(
                f"  FULL pair Jaccard     : {model_fp:.3f}  "
                f"(baseline {base_fp:.3f}, {_delta(model_fp, base_fp)})",
            )
        if run_members:
            model_mp = _mean([r.members_pair_jaccard for r in subset])
            base_mp = _mean([r.base_members_pair_jaccard for r in subset])
            lines.append(
                f"  MEMBERS-ONLY pair Jac : {model_mp:.3f}  "
                f"(baseline {base_mp:.3f}, {_delta(model_mp, base_mp)})",
            )

    metric_block("OVERALL", results)
    metric_block("COLD-START SLICE", cold)

    # --- explicit GATE verdict ---
    lines.append("=" * 72)
    if run_full:
        model_h1 = _mean([float(r.full_mentor_hit1) for r in results])
        base_h1 = _mean([float(r.base_mentor_hit1) for r in results])
        model_h3 = _mean([float(r.full_mentor_hit3) for r in results])
        base_h3 = _mean([float(r.base_mentor_hit3) for r in results])
        passed = model_h1 > base_h1 and model_h3 >= base_h3
        verdict = "PASS" if passed else "FAIL"
        lines.append(
            f"GATE A [{verdict}] — model must beat the most-frequent-mentor baseline: "
            f"hit@1 {model_h1:.3f} vs {base_h1:.3f}, hit@3 {model_h3:.3f} vs {base_h3:.3f}",
        )
    else:
        lines.append(
            "GATE A — FULL mode not run (--mode members); the mentor gate is FULL-only.",
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def _delta(model: float, base: float) -> str:
    diff = model - base
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f}"


def _resolve_model(value: str) -> Model:
    try:
        return Model(value)
    except ValueError as exc:
        valid = ", ".join(m.value for m in Model)
        msg = f"unknown --embedding-model {value!r}; valid values: {valid}"
        raise SystemExit(msg) from exc


async def main_async(ns: argparse.Namespace) -> int:
    model = _resolve_model(ns.embedding_model)
    weights = weights_from_args(ns)
    run_full = ns.mode in ("full", "both")
    run_members = ns.mode in ("members", "both")
    rng = random.Random(ns.seed)  # noqa: S311 - sample selection, not cryptography

    init_http_client()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()
    try:
        rows = await get_backtest_population(db, model)
        population = build_population(rows)
        if not population:
            print(
                "Backtest population is empty — ingest + embed defenses first "
                "(need status='Одбрана', a non-null embedding, mentor, and both members).",
                file=sys.stderr,
            )
            return 1

        global_top_mentor, mentor_cofreq = compute_baselines(population)
        global_prior = build_mentor_prior(
            (case.mentor, case.member1, case.member2) for case in population
        )
        # Only pay for paper retrieval when a paper-derived signal is actually on, so the
        # defenses-only GATE A path stays a pure (and fast) no-paper run.
        use_papers = (
            weights.expertise_weight > 0
            or weights.coauthor_weight > 0
            or weights.coauthor_member_boost > 0
        )
        # Global co-author prior (whole paper graph, built once; no leave-one-out — papers
        # are independent of the held-out defense's committee ground truth).
        coauthor_prior = (
            build_coauthor_prior(await get_all_paper_authors(db))
            if weights.coauthor_prior_weight > 0
            else None
        )
        evaluated = stratified_sample(population, ns.sample_size, rng)

        sem = asyncio.Semaphore(ns.concurrency)

        async def run_one(held: HeldOut) -> CaseResult | None:
            async with sem:
                try:
                    retrieved = await retrieve_similar_diplomas(
                        db,
                        held.title,
                        model,
                        ns.initial_k,
                        ns.top_k,
                        exclude_external_id=held.external_id,
                    )
                except Exception as exc:  # one bad case shouldn't abort the run
                    print(
                        f"  ! retrieval error on {held.external_id}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    return None
                if not retrieved:
                    return None

                expertise = EMPTY_EXPERTISE
                coauthors = EMPTY_COAUTHORS
                if use_papers:
                    try:
                        papers = await retrieve_professor_papers(
                            db,
                            held.title,
                            model,
                            ns.paper_k,
                        )
                    except Exception as exc:  # paper failure shouldn't abort the case
                        print(
                            f"  ! paper retrieval error on {held.external_id}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                        papers = []
                    if papers:
                        expertise = build_expertise_index(papers, weights)
                        coauthors = _accumulate_coauthor_edges(
                            papers,
                            weights,
                            now_year=ns.now_year,
                        )

                return evaluate_case(
                    held,
                    retrieved,
                    weights,
                    mentor_topk=ns.mentor_topk,
                    run_full=run_full,
                    run_members=run_members,
                    cold_start_floor=weights.cold_start_defense_floor,
                    global_top_mentor=global_top_mentor,
                    mentor_cofreq=mentor_cofreq,
                    global_prior=global_prior,
                    expertise=expertise,
                    coauthors=coauthors,
                    coauthor_prior=coauthor_prior,
                )

        gathered = await asyncio.gather(*(run_one(h) for h in evaluated))
        results = [r for r in gathered if r is not None]
    finally:
        await db.disconnect()
        await close_http_client()

    if not results:
        print(
            "No cases produced a result (all retrievals empty/failed).",
            file=sys.stderr,
        )
        return 1

    print(
        report(
            results,
            run_full=run_full,
            run_members=run_members,
            weights=weights,
            model_value=model.value,
            initial_k=ns.initial_k,
            top_k=ns.top_k,
            mentor_topk=ns.mentor_topk,
            population_size=len(population),
        ),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Committee-recommender backtest (GATE A: defenses-only).",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="evaluate ~N cases, stratified by mentor (0 = the full population)",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="headline run over the full population (equivalent to --sample-size 0)",
    )
    p.add_argument(
        "--mode",
        choices=("full", "members", "both"),
        default="both",
        help="which mode(s) to evaluate (default both)",
    )
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDINGS_MODEL.value)
    p.add_argument("--initial-k", type=int, default=30, help="ANN candidate budget")
    p.add_argument("--top-k", type=int, default=10, help="reranked candidates kept")
    p.add_argument(
        "--paper-k",
        type=int,
        default=50,
        help="paper KNN breadth for the expertise/buddy signals (GATE B)",
    )
    p.add_argument(
        "--mentor-topk",
        type=int,
        default=ScoringWeights().mentor_topk,
        help="mentor candidate breadth for the FULL argmax",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234, help="stratified-sample RNG seed")

    # --- ScoringWeights overrides (None => keep the dataclass default) ---
    p.add_argument("--similarity-weight", type=float, default=None)
    p.add_argument("--rerank-weight", type=float, default=None)
    p.add_argument("--recency-half-life-days", type=float, default=None)
    p.add_argument("--pair-affinity-weight", type=float, default=None)
    p.add_argument("--mentor-prior-weight", type=float, default=None)
    p.add_argument("--coauthor-prior-weight", type=float, default=None)
    # Off-by-default at GATE A; present so the script is forward-compatible with papers.
    p.add_argument("--expertise-weight", type=float, default=None)
    p.add_argument("--coauthor-weight", type=float, default=None)
    p.add_argument("--coauthor-member-boost", type=float, default=None)
    p.add_argument("--coauthor-recency-half-life-days", type=float, default=None)
    p.add_argument(
        "--now-year",
        type=int,
        default=None,
        help="reference year for buddy recency decay (e.g. 2026); None = recency off",
    )

    # --- ablation switches (force a signal to 0; no-ops at GATE A) ---
    p.add_argument(
        "--no-papers",
        action="store_true",
        help="force expertise_weight=0 (defenses-only; the GATE A default)",
    )
    p.add_argument(
        "--no-buddies",
        action="store_true",
        help="force coauthor_weight=0 and coauthor_member_boost=0 (composes with --no-papers)",
    )
    return p


def _resolve_sample_size(ns: argparse.Namespace) -> None:
    if ns.full:
        ns.sample_size = 0


if __name__ == "__main__":
    namespace = build_parser().parse_args()
    _resolve_sample_size(namespace)
    raise SystemExit(asyncio.run(main_async(namespace)))
