"""cd api && uv run python ../scripts/backtest.py --sample-size 300"""

# ruff: noqa: INP001 - standalone offline script; scripts/ is not an importable package

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

if (Path.cwd() / "app").is_dir() and str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

from asyncpg import Record

from app.constants.defaults import DEFAULT_EMBEDDINGS_MODEL
from app.data import (
    get_active_staff_names,
    get_mentor_eligible_names,
    get_staff_aliases,
)
from app.data.connection import Database
from app.data.diplomas import get_backtest_population
from app.data.professor_documents import get_all_paper_authors
from app.llms.diploma_retrieval import retrieve_similar_diplomas
from app.llms.embeddings import generate_embeddings
from app.llms.models import Model
from app.llms.professor_retrieval import retrieve_professor_papers
from app.llms.text_utils import _prepare_text_for_embedding
from app.recommenders.config import ScoringWeights
from app.recommenders.recommend import (
    CoauthorIndex,
    ExpertiseIndex,
    MentorPriorIndex,
    Mode,
    RetrievedDiploma,
    RetrievedPaper,
    SelectionConstraints,
    _accumulate_coauthor_edges,
    build_coauthor_prior,
    build_expertise_index,
    build_mentor_prior,
    build_recent_member_prior,
    score_people,
    select_committee,
)
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

settings = Settings()

EMPTY_EXPERTISE = ExpertiseIndex(by_professor={}, supporting={})
EMPTY_COAUTHORS = CoauthorIndex(edges={}, supporting={})


@dataclass(frozen=True)
class HeldOut:
    external_id: str
    title: str
    mentor: str
    member1: str
    member2: str
    submitted: date | None = None

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
            submitted=row["date_of_submission"],
        )
        for row in rows
    ]


def stratified_sample(
    population: list[HeldOut],
    sample_size: int,
    rng: random.Random,
) -> list[HeldOut]:
    if sample_size <= 0 or sample_size >= len(population):
        return list(population)

    by_mentor: dict[str, list[HeldOut]] = defaultdict(list)
    for case in population:
        by_mentor[case.mentor].append(case)

    total = len(population)
    sampled: list[HeldOut] = []
    for cases in by_mentor.values():
        quota = max(1, round(sample_size * len(cases) / total))
        quota = min(quota, len(cases))
        sampled.extend(rng.sample(cases, quota))

    rng.shuffle(sampled)
    return sampled[:sample_size]


def weights_from_args(ns: argparse.Namespace) -> ScoringWeights:
    base = ScoringWeights()

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
    # Without LOO, the held-out defense leaks its own members into its mentor's prior bucket.
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

    base_hit1 = global_top_mentor == held.mentor
    base_hit3 = (
        global_top_mentor == held.mentor
    )  # single-name baseline -> same as hit@1
    # LOO for the baseline: subtract the held-out defense's members from its mentor's counts to avoid leaking the true pair into the baseline.
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
    lines: list[str] = [
        "=" * 72,
        "COMMITTEE-RECOMMENDER BACKTEST — GATE A (defenses-only)",
        "=" * 72,
        f"model={model_value}  population={population_size}  evaluated={n}  "
        f"cold_start={len(cold)}",
        f"initial_k={initial_k}  top_k={top_k}  mentor_topk={mentor_topk}  "
        f"reranker_min={settings.DIPLOMA_RERANKER_MIN_SCORE}",
        f"weights: sim={weights.similarity_weight} rerank={weights.rerank_weight} "
        f"recency_hl_days={weights.recency_half_life_days} "
        f"pair_affinity={weights.pair_affinity_weight} "
        f"mentor_prior={weights.mentor_prior_weight} "
        f"coauthor_prior={weights.coauthor_prior_weight} "
        f"expertise={weights.expertise_weight} coauthor={weights.coauthor_weight}",
    ]

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
    # Sample selection, not cryptography.
    rng = random.Random(ns.seed)  # noqa: S311

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
        use_papers = (
            weights.expertise_weight > 0
            or weights.coauthor_weight > 0
            or weights.coauthor_member_boost > 0
        )
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


@dataclass(frozen=True)
class TimedCase:
    held: HeldOut
    submitted: date
    retrieved: list[RetrievedDiploma]
    papers: list[RetrievedPaper]


@dataclass(frozen=True)
class TimedScore:
    recommended: bool
    mentor_hit1: bool
    shortlist_hit: bool
    members_jaccard: float


type Committee = tuple[str, str, str, date]


def _canonical_held(held: HeldOut, aliases: Mapping[str, str]) -> HeldOut:
    return replace(
        held,
        mentor=aliases.get(held.mentor, held.mentor),
        member1=aliases.get(held.member1, held.member1),
        member2=aliases.get(held.member2, held.member2),
    )


def _canonical_retrieved(
    diploma: RetrievedDiploma,
    aliases: Mapping[str, str],
) -> RetrievedDiploma:
    return replace(
        diploma,
        mentor=aliases.get(diploma.mentor, diploma.mentor),
        member1=aliases.get(diploma.member1, diploma.member1)
        if diploma.member1
        else None,
        member2=aliases.get(diploma.member2, diploma.member2)
        if diploma.member2
        else None,
    )


def score_timed_case(
    case: TimedCase,
    weights: ScoringWeights,
    committees: Sequence[Committee],
    *,
    active: frozenset[str],
    eligible: frozenset[str],
    use_history: bool,
    member_years: int,
) -> TimedScore:
    """Score one defense using only what was known before its date."""
    as_of = case.submitted
    earlier = [row for row in committees if row[3] < as_of]
    profile = (
        weights
        if use_history
        else replace(weights, mentor_prior_weight=0.0, pair_affinity_weight=0.0)
    )
    mentor_prior = (
        build_mentor_prior(row[:3] for row in earlier) if use_history else None
    )
    member_prior = (
        build_recent_member_prior(earlier, as_of, member_years)
        if use_history and member_years > 0
        else None
    )
    expertise = build_expertise_index(case.papers, profile, now_year=as_of.year)

    recommended = mentor_hit1 = shortlist_hit = False
    if case.retrieved:
        ranked = score_people(
            case.retrieved,
            expertise,
            None,
            profile,
            Mode.FULL,
            now=as_of,
        )
        full = select_committee(
            ranked,
            Mode.FULL,
            None,
            mentor_topk=3,
            mentor_prior=mentor_prior,
            member_prior=member_prior,
            constraints=SelectionConstraints(
                allowed=active,
                mentor_allowed=eligible,
                alternative_count=3,
                distinct_mentors=True,
            ),
        )
        recommended = full.mentor is not None and len(full.members) == 2
        mentor_hit1 = full.mentor == case.held.mentor
        shortlist_hit = case.held.mentor in {
            alternative.mentor for alternative in full.alternatives
        }

    ranked_members = score_people(
        case.retrieved,
        expertise,
        None,
        profile,
        Mode.MEMBERS_ONLY,
        given_mentor=case.held.mentor,
        now=as_of,
    )
    members = select_committee(
        ranked_members,
        Mode.MEMBERS_ONLY,
        case.held.mentor,
        mentor_topk=3,
        mentor_prior=mentor_prior,
        member_prior=member_prior,
        constraints=SelectionConstraints(allowed=active, mentor_allowed=eligible),
    )
    return TimedScore(
        recommended,
        mentor_hit1,
        shortlist_hit,
        _jaccard(members.members, case.held.true_pair),
    )


def _bootstrap_interval(
    deltas: Sequence[float],
    rng: random.Random,
    rounds: int = 2000,
) -> tuple[float, float]:
    size = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=size)) / size for _ in range(rounds))
    return means[int(rounds * 0.025)], means[int(rounds * 0.975)]


def _timed_line(label: str, scores: Sequence[TimedScore]) -> str:
    return (
        f"{label:24} recs={_mean([float(s.recommended) for s in scores]):.3f} "
        f"hit1={_mean([float(s.mentor_hit1) for s in scores]):.3f} "
        f"shortlist={_mean([float(s.shortlist_hit) for s in scores]):.3f} "
        f"members={_mean([s.members_jaccard for s in scores]):.3f}"
    )


async def _load_timed_case(
    db: Database,
    held: HeldOut,
    model: Model,
    ns: argparse.Namespace,
    aliases: Mapping[str, str],
) -> TimedCase | None:
    if held.submitted is None:
        return None
    try:
        embedding = await generate_embeddings(
            _prepare_text_for_embedding(held.title, model, is_document=False),
            model,
        )
        retrieved = await retrieve_similar_diplomas(
            db,
            held.title,
            model,
            ns.initial_k,
            ns.top_k,
            exclude_external_id=held.external_id,
            query_embedding=embedding,
            submitted_before=held.submitted,
        )
        papers = await retrieve_professor_papers(
            db,
            held.title,
            model,
            ns.paper_k,
            query_embedding=embedding,
            published_through=held.submitted.year,
        )
    except Exception as exc:  # one bad case shouldn't abort the run
        print(f"  ! retrieval error: {type(exc).__name__}", file=sys.stderr)
        return None
    return TimedCase(
        held,
        held.submitted,
        [_canonical_retrieved(diploma, aliases) for diploma in retrieved],
        [
            replace(
                paper,
                coauthors=tuple(
                    dict.fromkeys(aliases.get(name, name) for name in paper.coauthors),
                ),
            )
            for paper in papers
        ],
    )


async def time_aware_async(ns: argparse.Namespace) -> int:
    """Replay production staff rules without letting a case see its future."""
    model = _resolve_model(ns.embedding_model)
    weights = weights_from_args(ns)
    rng = random.Random(ns.seed)  # noqa: S311
    init_http_client()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()
    try:
        aliases = await get_staff_aliases()
        active = await get_active_staff_names()
        eligible = await get_mentor_eligible_names()
        population = [
            _canonical_held(held, aliases)
            for held in build_population(await get_backtest_population(db, model))
        ]
        committees: list[Committee] = [
            (held.mentor, held.member1, held.member2, held.submitted)
            for held in population
            if held.submitted is not None
        ]
        selected = [
            held
            for held in population
            if held.submitted is not None
            and ns.from_date <= held.submitted < ns.to_date
            and held.mentor in eligible
            and {held.member1, held.member2} <= active
        ]
        if 0 < ns.sample_size < len(selected):
            selected = rng.sample(selected, ns.sample_size)

        semaphore = asyncio.Semaphore(ns.concurrency)

        async def load(held: HeldOut) -> TimedCase | None:
            async with semaphore:
                return await _load_timed_case(db, held, model, ns, aliases)

        loaded = await asyncio.gather(*(load(held) for held in selected))
    finally:
        await db.disconnect()
        await close_http_client()

    cases = [case for case in loaded if case is not None]
    if not cases:
        print("No time-aware cases produced a result.", file=sys.stderr)
        return 1

    def run(*, use_history: bool, member_years: int) -> list[TimedScore]:
        return [
            score_timed_case(
                case,
                weights,
                committees,
                active=active,
                eligible=eligible,
                use_history=use_history,
                member_years=member_years,
            )
            for case in cases
        ]

    print(
        f"TIME-AWARE BACKTEST {ns.from_date}..{ns.to_date} cases={len(cases)} "
        f"failed={len(selected) - len(cases)} "
        f"floor={settings.DIPLOMA_RERANKER_MIN_SCORE}",
    )
    print(_timed_line("planner (no history)", run(use_history=False, member_years=0)))
    chat = run(use_history=True, member_years=0)
    print(_timed_line("chat (all history)", chat))
    for years in ns.member_prior_years:
        scores = run(use_history=True, member_years=years)
        deltas = [
            score.members_jaccard - base.members_jaccard
            for score, base in zip(scores, chat, strict=True)
        ]
        low, high = _bootstrap_interval(deltas, rng)
        print(
            f"{_timed_line(f'chat, members {years}y', scores)} "
            f"delta={_mean(deltas):+.3f} ci95=[{low:+.3f},{high:+.3f}]",
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

    p.add_argument("--similarity-weight", type=float, default=None)
    p.add_argument("--rerank-weight", type=float, default=None)
    p.add_argument("--recency-half-life-days", type=float, default=None)
    p.add_argument("--pair-affinity-weight", type=float, default=None)
    p.add_argument("--mentor-prior-weight", type=float, default=None)
    p.add_argument("--coauthor-prior-weight", type=float, default=None)
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
    p.add_argument(
        "--time-aware",
        action="store_true",
        help="replay production staff rules using only data from before each defense",
    )
    p.add_argument("--from-date", type=date.fromisoformat, default=date(2025, 7, 1))
    p.add_argument("--to-date", type=date.fromisoformat, default=date.max)
    p.add_argument(
        "--member-prior-years",
        type=int,
        nargs="*",
        default=[2, 3, 4, 5],
        help="recent-history windows to compare for member selection",
    )
    return p


def _resolve_sample_size(ns: argparse.Namespace) -> None:
    if ns.full:
        ns.sample_size = 0


if __name__ == "__main__":
    namespace = build_parser().parse_args()
    _resolve_sample_size(namespace)
    entrypoint = time_aware_async if namespace.time_aware else main_async
    raise SystemExit(asyncio.run(entrypoint(namespace)))
