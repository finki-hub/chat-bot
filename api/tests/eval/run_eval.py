"""Retrieval-evaluation harness for the FINKI RAG pipeline.

Runs a golden set of (query -> expected source) examples through the *real* retrieval
stack and measures recall at two stages, so a threshold/model/prompt change can be
A/B'd objectively instead of by hand. It deliberately reuses the production building
blocks (`build_query_variants`, `_embed_variant`, `get_closest_*`, the cross-encoder
rerank) so the numbers reflect what `/chat` actually retrieves.

Two recalls are reported per example, and their gap is the headline signal:

* ANN recall (ideal)   - is the expected source in the top-N by raw cosine distance,
                         with the distance ceiling disabled? (limit-only)
* ANN recall (prod)    - is it in the candidate pool the production threshold actually
                         lets through? (`MODEL_DISTANCE_THRESHOLDS`)
* final recall@top_k   - is it in the reranked, min-score-filtered context?

The prod-miss gap (ideal-hit but absent from the prod pool) is split into `ceiling drops`
(past the per-model distance cutoff — the "praksa" class: a relevant source silently
discarded before the reranker sees it) and `k-budget drops` (crowded out of the
per-query-k pool). A healthy config keeps ceiling drops at zero.

Run it inside the api container (has DATABASE_URL, GPU/OpenAI access, app code):

    docker exec -e PYTHONPATH=/app finki-hub-chat-bot-api-1 \
        python /app/tests/eval/run_eval.py \
        --golden /app/tests/eval/golden.jsonl \
        --embedding-model BAAI/bge-m3

Compare embedders by re-running with `--embedding-model text-embedding-3-large`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from asyncpg import PostgresError

from app.data.connection import Database
from app.data.documents import get_closest_chunks
from app.data.questions import get_closest_questions, get_matching_questions
from app.llms.context import (
    _build_candidates,
    _Candidate,
    _embed_variant,
    _select_with_source_priority,
    _valid_rerank_results,
)
from app.llms.faq_evidence_constraints import (
    apply_faq_constraints_to_candidates,
    faq_constraints_filter_evidence,
    plan_faq_evidence_constraints,
)
from app.llms.faq_query_recovery_plan import (
    FaqQueryRecoveryOutcome,
    plan_faq_query_recovery,
)
from app.llms.models import MODEL_DISTANCE_THRESHOLDS, Model
from app.llms.query_normalization import query_search_variants
from app.llms.query_variants import build_query_variants
from app.llms.reranker import post_rerank as _post_rerank
from app.llms.retrieval_budget import retrieval_budget
from app.llms.retrieval_routing import lexical_faq_expansion_allowed
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from app.utils.http_client import close_http_client, init_http_client
from app.utils.settings import Settings

settings = Settings()


# A ceiling high enough to disable the distance pre-filter (cosine distance maxes at 2.0).
NO_THRESHOLD = 9.0

type FaqKey = tuple[Literal["Q"], str]
type ChunkKey = tuple[Literal["C"], str, int]
type AbstainKey = tuple[Literal["none"]]
type NaturalKey = FaqKey | ChunkKey | AbstainKey


@dataclass
class Example:
    id: str
    query: str
    anchor: Mapping[str, str | int]
    category: str = ""
    difficulty: str = ""
    cohort: str = ""

    @property
    def is_abstain(self) -> bool:
        return self.anchor.get("type") == "none"


def anchor_key(anchor: Mapping[str, str | int]) -> NaturalKey:
    """Stable natural key for an anchor, used to test membership in retrieved results."""
    if anchor["type"] == "Q":
        return ("Q", str(anchor["name"]))
    if anchor["type"] == "C":
        return ("C", str(anchor["document_name"]), int(anchor["chunk_index"]))
    return ("none",)


def question_key(q: QuestionSchema) -> FaqKey:
    return ("Q", q.name)


def chunk_key(c: ChunkSchema) -> ChunkKey:
    return ("C", c.document_name, int(c.chunk_index))


def final_context_hit(
    ex: Example,
    want: NaturalKey,
    topk: list[NaturalKey],
) -> tuple[bool, int | None]:
    if ex.is_abstain:
        return bool(topk), 1 if topk else None
    for i, nat in enumerate(topk):
        if nat == want:
            return True, i + 1
    return False, None


@dataclass
class Result:
    example: Example
    execution_status: Literal["completed", "error"] = "completed"
    error_type: str | None = None
    ann_ideal: bool = False
    ann_prod: bool = False
    final: bool = False
    rank: int | None = None
    n_candidates: int = 0
    candidate_hit: bool = False
    faq_evidence_constraint_applied: bool = False
    faq_evidence_constraint_reason: str = "none"
    faq_evidence_excluded_candidate_count: int = 0
    faq_evidence_constrained_abstained: bool = False
    local_faq_recovery_outcome: FaqQueryRecoveryOutcome = "skipped"
    local_query_variant_added: bool = False
    local_rerank_query_used: bool = False
    best_distance: float | None = None
    effective_initial_k: int | None = None
    effective_per_query_k: int | None = None


@dataclass
class Aggregate:
    results: list[Result] = field(default_factory=list)
    model_threshold: float = 0.5

    def _subset(self, pred) -> list[Result]:
        return [
            r for r in self.results if r.execution_status == "completed" and pred(r)
        ]

    def report(self) -> str:
        retr = self._subset(lambda r: not r.example.is_abstain)
        abst = self._subset(lambda r: r.example.is_abstain)
        errors = [r for r in self.results if r.execution_status == "error"]
        lines = [
            (
                "EXECUTION: "
                f"total={len(self.results)} completed={len(self.results) - len(errors)} "
                f"errors={len(errors)}"
            ),
        ]
        if errors:
            lines.append(
                "INVALID RUN: quality metrics below cover completed cases only."
            )
            lines.extend(
                f"  [EXECUTION-ERROR] {r.example.id}: {r.error_type}" for r in errors
            )

        def block(title: str, rs: list[Result]) -> None:
            if not rs:
                return
            n = len(rs)
            ann_ideal = sum(r.ann_ideal for r in rs)
            ann_prod = sum(r.ann_prod for r in rs)
            final = sum(r.final for r in rs)
            prod_miss = [r for r in rs if r.ann_ideal and not r.ann_prod]
            ceiling = sum(
                1
                for r in prod_miss
                if r.best_distance is not None
                and r.best_distance >= self.model_threshold
            )
            kbudget = len(prod_miss) - ceiling
            mrr = sum((1.0 / r.rank) for r in rs if r.rank) / n
            lines.append(f"  {title} (n={n})")
            lines.append(
                f"    ANN recall  ideal : {ann_ideal}/{n}  ({100 * ann_ideal / n:.1f}%)",
            )
            lines.append(
                f"    ANN recall  prod  : {ann_prod}/{n}  ({100 * ann_prod / n:.1f}%)",
            )
            lines.append(
                f"    ceiling drops     : {ceiling}   <- relevant but past the distance cutoff (praksa class)",
            )
            lines.append(
                f"    k-budget drops    : {kbudget}   <- relevant but crowded out of the per-query-k pool",
            )
            lines.append(
                f"    final recall@k    : {final}/{n}  ({100 * final / n:.1f}%)",
            )
            lines.append(f"    MRR (reranked)    : {mrr:.3f}")

        lines.append("=" * 64)
        lines.append("RETRIEVAL EXAMPLES")
        block("overall", retr)
        block(
            "source=faq",
            self._subset(
                lambda r: not r.example.is_abstain and r.example.anchor["type"] == "Q",
            ),
        )
        block(
            "source=chunk",
            self._subset(
                lambda r: not r.example.is_abstain and r.example.anchor["type"] == "C",
            ),
        )
        block(
            "difficulty=easy",
            self._subset(
                lambda r: not r.example.is_abstain and r.example.difficulty == "easy",
            ),
        )
        block(
            "difficulty=hard",
            self._subset(
                lambda r: not r.example.is_abstain and r.example.difficulty == "hard",
            ),
        )
        for cohort in sorted({r.example.cohort for r in retr if r.example.cohort}):
            block(
                f"cohort={cohort}",
                self._subset(
                    lambda r, cohort=cohort: (
                        not r.example.is_abstain and r.example.cohort == cohort
                    ),
                ),
            )

        if abst:
            n = len(abst)
            leaked = sum(1 for r in abst if r.final)
            lines.append("-" * 64)
            lines.append("ABSTAIN / OUT-OF-SCOPE EXAMPLES")
            lines.append(
                f"  n={n}; retrieved a (false) top-k source for {leaked}/{n} "
                "(lower is better — these should ideally find nothing relevant)",
            )

        lines.append("=" * 64)
        lines.append("MISSES (final recall failures on retrieval examples):")
        for r in retr:
            if not r.final:
                if not r.ann_ideal:
                    tag = "ANN-MISS"
                elif not r.ann_prod:
                    tag = (
                        "CEILING-DROP"
                        if r.best_distance is not None
                        and r.best_distance >= self.model_threshold
                        else "KBUDGET-DROP"
                    )
                else:
                    tag = "RERANK-MISS"
                lines.append(
                    f"  [{tag}] {r.example.id} ({r.example.difficulty}/{r.example.category}) "
                    f"best_dist={r.best_distance}",
                )
                lines.append(f"      q: {r.example.query}")
                lines.append(f"      want: {anchor_key(r.example.anchor)}")
        return "\n".join(lines)


async def _search_pair(
    db: Database,
    embedding: list[float],
    model: Model,
    *,
    limit: int,
    threshold: float | None = None,
) -> tuple[list[QuestionSchema], list[ChunkSchema]]:
    """Keep ANN diagnostic thresholds while draining sibling reads on failure."""
    faq_task = asyncio.create_task(
        get_closest_questions(db, embedding, model, limit=limit, threshold=threshold),
    )
    chunk_task = asyncio.create_task(
        get_closest_chunks(db, embedding, model, limit=limit, threshold=threshold),
    )
    try:
        questions, chunks = await asyncio.gather(faq_task, chunk_task)
    except BaseException:
        faq_task.cancel()
        chunk_task.cancel()
        await asyncio.gather(faq_task, chunk_task, return_exceptions=True)
        raise
    return questions, chunks


async def evaluate_one(
    db: Database,
    ex: Example,
    *,
    embedding_model: Model,
    initial_k: int,
    top_k: int,
    ideal_limit: int,
) -> Result:
    want = anchor_key(ex.anchor)
    evidence_plan = await plan_faq_evidence_constraints(db, ex.query)
    if evidence_plan.outcome == "error":
        raise ValueError("FAQ evidence constraint catalog failure")
    constraints = evidence_plan.constraints

    variant_bundle = build_query_variants(ex.query)
    variants = [
        (variant.text, variant.is_document) for variant in variant_bundle.variants
    ]
    budget = retrieval_budget(initial_k, variant_count=len(variants))
    recovery_plan = await plan_faq_query_recovery(db, ex.query)
    canonical_query = (
        recovery_plan.recovery.canonical_query
        if recovery_plan.recovery is not None
        else None
    )
    local_variant_added = (
        canonical_query is not None
        and canonical_query.casefold().rstrip("?!. ").split()
        != ex.query.casefold().rstrip("?!. ").split()
    )
    per_query_k = (
        min(budget.per_query_k, 31)
        if recovery_plan.outcome == "matched"
        else budget.per_query_k
    )
    ideal_probe_limit = max(ideal_limit, per_query_k)

    embeddings = await asyncio.gather(
        *(
            _embed_variant(text, embedding_model, is_document=is_doc)
            for text, is_doc in variants
        ),
    )
    mandatory_embedding_count = len(embeddings)
    if local_variant_added and canonical_query is not None:
        try:
            canonical_embedding = await _embed_variant(
                canonical_query,
                embedding_model,
                is_document=False,
            )
        except Exception:  # ruff: ignore[BLE001] -- same optional canonical-embedding degradation as production
            local_variant_added = False
        else:
            embeddings.append(canonical_embedding)

    # Mandatory searches retain their existing error semantics. Only the extra
    # local canonical search is optional, as in production.
    prod_lists = await asyncio.gather(
        *(
            _search_pair(db, emb, embedding_model, limit=per_query_k)
            for emb in embeddings[:mandatory_embedding_count]
        ),
    )
    if local_variant_added:
        try:
            canonical_results = await _search_pair(
                db,
                embeddings[-1],
                embedding_model,
                limit=per_query_k,
            )
        except Exception:  # ruff: ignore[BLE001] -- only optional canonical search degrades; cancellation propagates
            local_variant_added = False
            embeddings.pop()
        else:
            prod_lists.append(canonical_results)

    # --- ANN ideal (no threshold) ---
    ideal_lists = await asyncio.gather(
        *(
            _search_pair(
                db,
                emb,
                embedding_model,
                limit=ideal_probe_limit,
                threshold=NO_THRESHOLD,
            )
            for emb in embeddings
        ),
    )
    ann_ideal = False
    best_distance: float | None = None
    for qs, cs in ideal_lists:
        for q in qs:
            if question_key(q) == want:
                ann_ideal = True
                if q.distance is not None:
                    best_distance = (
                        q.distance
                        if best_distance is None
                        else min(best_distance, q.distance)
                    )
        for c in cs:
            if chunk_key(c) == want:
                ann_ideal = True
                if c.distance is not None:
                    best_distance = (
                        c.distance
                        if best_distance is None
                        else min(best_distance, c.distance)
                    )

    # --- Build the production rerank candidate pool ---
    lexical_lists: list[list[QuestionSchema]] = []
    lexical_queries = query_search_variants(ex.query)
    if recovery_plan.outcome != "matched" and lexical_faq_expansion_allowed(
        (
            question.distance
            for questions, _chunks in prod_lists
            for question in questions
        ),
        (chunk.distance for _questions, chunks in prod_lists for chunk in chunks),
        has_transliterated_query=len(lexical_queries) > 1,
    ):
        try:
            lexical_lists = await asyncio.gather(
                *(
                    get_matching_questions(db, query, limit=per_query_k)
                    for query in lexical_queries
                ),
            )
        except PostgresError:
            lexical_lists = []

    natural_by_key: dict[str, NaturalKey] = {}
    ann_prod = False
    for qs, cs in prod_lists:
        for q in qs:
            natural_by_key[f"Q:{q.id}"] = question_key(q)
            if question_key(q) == want:
                ann_prod = True
        for ch in cs:
            natural_by_key[f"C:{ch.id}"] = chunk_key(ch)
            if chunk_key(ch) == want:
                ann_prod = True

    for questions in lexical_lists:
        for question in questions:
            natural_by_key[f"Q:{question.id}"] = question_key(question)
    direct_faq = recovery_plan.question if recovery_plan.outcome == "matched" else None
    if direct_faq is not None:
        natural_by_key[f"Q:{direct_faq.id}"] = question_key(direct_faq)
    candidates = _build_candidates(prod_lists, lexical_lists, direct_faq)
    pre_constraint_count = len(candidates)
    constrained_abstained = False
    strict_constraints = (
        constraints
        if constraints is not None and faq_constraints_filter_evidence(constraints)
        else None
    )
    if constraints is not None:
        candidates = apply_faq_constraints_to_candidates(candidates, constraints)
        constrained_abstained = pre_constraint_count > 0 and not candidates
    cand_keys = [candidate.key for candidate in candidates]
    cand_rerank = [candidate.rerank_text for candidate in candidates]
    cand_nat = [natural_by_key[key] for key in cand_keys]

    final_hit = False
    rank: int | None = None
    if cand_rerank:
        try:
            response = await _post_rerank(
                {
                    "query": canonical_query or variant_bundle.rerank_query,
                    "documents": cand_rerank,
                },
            )
            ranked, _invalid_count = _valid_rerank_results(
                response.json(), len(candidates)
            )
        except Exception:  # noqa: BLE001 -- constrained production path abstains on reranker failure
            if strict_constraints is None:
                raise
            ranked = []
        kept_candidates: list[_Candidate] = []
        for idx, score in ranked:
            if score < settings.RERANKER_MIN_SCORE:
                continue
            kept_candidates.append(candidates[idx])
        if (
            not kept_candidates and ranked and strict_constraints is None
        ):  # production keeps the single best if all below floor
            best_idx = ranked[0][0]
            if 0 <= best_idx < len(cand_nat):
                kept_candidates = [candidates[best_idx]]
        if strict_constraints is not None and not kept_candidates:
            constrained_abstained = True
        natural_key_by_candidate = dict(zip(cand_keys, cand_nat, strict=True))
        topk = [
            natural_key_by_candidate[candidate.key]
            for candidate in _select_with_source_priority(kept_candidates, top_k)
        ]
        final_hit, rank = final_context_hit(ex, want, topk)

    return Result(
        example=ex,
        ann_ideal=ann_ideal,
        ann_prod=ann_prod,
        final=final_hit,
        rank=rank,
        n_candidates=len(cand_keys),
        faq_evidence_constraint_applied=constraints is not None,
        faq_evidence_constraint_reason=constraints.reason if constraints else "none",
        faq_evidence_excluded_candidate_count=pre_constraint_count - len(candidates),
        faq_evidence_constrained_abstained=constrained_abstained,
        candidate_hit=bool(cand_keys) if ex.is_abstain else want in cand_nat,
        local_faq_recovery_outcome=recovery_plan.outcome,
        local_query_variant_added=local_variant_added,
        local_rerank_query_used=recovery_plan.recovery is not None
        and bool(cand_rerank),
        best_distance=round(best_distance, 4) if best_distance is not None else None,
        effective_initial_k=budget.initial_k,
        effective_per_query_k=per_query_k,
    )


def execution_error_type(exc: Exception) -> str:
    """Keep a bounded class label, never an exception message or representation."""
    return re.sub(r"[^A-Za-z0-9_]", "_", type(exc).__name__[:80]) or "Exception"


def load_golden(path: Path) -> list[Example]:
    examples: list[Example] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        examples.append(
            Example(
                id=row["id"],
                query=row["query"],
                anchor=row["anchor"],
                category=row.get("category", ""),
                difficulty=row.get("difficulty", ""),
                cohort=row.get("cohort", ""),
            ),
        )
    return examples


async def main_async(ns: argparse.Namespace) -> int:
    examples = load_golden(Path(ns.golden))
    if ns.limit:
        examples = examples[: ns.limit]
    embedding_model = Model(ns.embedding_model)

    init_http_client()
    db = Database(dsn=settings.DATABASE_URL)
    await db.init()

    agg = Aggregate(model_threshold=MODEL_DISTANCE_THRESHOLDS.get(embedding_model, 0.5))
    try:
        sem = asyncio.Semaphore(ns.concurrency)

        async def run(ex: Example) -> Result:
            async with sem:
                try:
                    return await evaluate_one(
                        db,
                        ex,
                        embedding_model=embedding_model,
                        initial_k=ns.initial_k,
                        top_k=ns.top_k,
                        ideal_limit=ns.ideal_limit,
                    )
                # ruff: ignore[BLE001] -- one failed case must not abort the evaluation run
                except Exception as exc:
                    error_type = execution_error_type(exc)
                    print(
                        f"  ! execution error on {ex.id}: {error_type}",
                        file=sys.stderr,
                    )
                    return Result(
                        example=ex,
                        execution_status="error",
                        error_type=error_type,
                    )

        agg.results = await asyncio.gather(*(run(ex) for ex in examples))
    finally:
        await db.disconnect()
        await close_http_client()

    header = (
        f"golden={ns.golden}  n={len(examples)}  embed={embedding_model.value}  "
        f"requested_initial_k={ns.initial_k}  "
        f"ideal_limit_floor={ns.ideal_limit}  top_k={ns.top_k}  "
        f"reranker_min={settings.RERANKER_MIN_SCORE}"
    )
    print(header)
    print(agg.report())

    if ns.json:
        requested_budget = retrieval_budget(ns.initial_k)
        out = {
            "config": {
                "embedding_model": embedding_model.value,
                "initial_k": requested_budget.initial_k,
                "per_query_k": requested_budget.per_query_k,
                "ideal_limit": max(ns.ideal_limit, requested_budget.per_query_k),
                "requested_initial_k": ns.initial_k,
                "ideal_limit_floor": ns.ideal_limit,
                "top_k": ns.top_k,
                "reranker_min_score": settings.RERANKER_MIN_SCORE,
            },
            "results": [
                {
                    "id": r.example.id,
                    "execution_status": r.execution_status,
                    "error_type": r.error_type,
                    "difficulty": r.example.difficulty,
                    "category": r.example.category,
                    "cohort": r.example.cohort,
                    "anchor": r.example.anchor,
                    "ann_ideal": r.ann_ideal,
                    "ann_prod": r.ann_prod,
                    "final": r.final,
                    "rank": r.rank,
                    "n_candidates": r.n_candidates
                    if r.execution_status == "completed"
                    else None,
                    "candidate_hit": r.candidate_hit
                    if r.execution_status == "completed"
                    else None,
                    "local_faq_recovery_outcome": r.local_faq_recovery_outcome
                    if r.execution_status == "completed"
                    else None,
                    "local_query_variant_added": r.local_query_variant_added
                    if r.execution_status == "completed"
                    else None,
                    "local_rerank_query_used": r.local_rerank_query_used
                    if r.execution_status == "completed"
                    else None,
                    "best_distance": r.best_distance,
                    "faq_evidence_constraint_applied": r.faq_evidence_constraint_applied
                    if r.execution_status == "completed"
                    else None,
                    "faq_evidence_constraint_reason": r.faq_evidence_constraint_reason
                    if r.execution_status == "completed"
                    else None,
                    "faq_evidence_excluded_candidate_count": r.faq_evidence_excluded_candidate_count
                    if r.execution_status == "completed"
                    else None,
                    "faq_evidence_constrained_abstained": r.faq_evidence_constrained_abstained
                    if r.execution_status == "completed"
                    else None,
                    "effective_initial_k": r.effective_initial_k,
                    "effective_per_query_k": r.effective_per_query_k,
                }
                for r in agg.results
            ],
        }
        Path(ns.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote per-example JSON to {ns.json}")

    return 2 if any(r.execution_status == "error" for r in agg.results) else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FINKI retrieval-evaluation harness")
    p.add_argument("--golden", default=str(Path(__file__).parent / "golden.jsonl"))
    p.add_argument("--embedding-model", default=Model.BGE_M3_LOCAL.value)
    p.add_argument(
        "--initial-k",
        type=int,
        default=30,
        help="total ANN budget; per-variant k = initial_k//variant_count + 1",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--ideal-limit",
        type=int,
        default=20,
        help="per-variant limit for the no-threshold ideal-recall probe",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="evaluate only the first N examples (0 = all); for smoke tests",
    )
    p.add_argument(
        "--json",
        default=None,
        help="optional path to write per-example results as JSON",
    )
    return p


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(build_parser().parse_args())))
