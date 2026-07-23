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
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from app.data.connection import Database
from app.data.documents import get_closest_chunks
from app.data.questions import get_closest_questions
from app.llms.context import (
    _Candidate,
    _chunk_candidate,
    _embed_variant,
    _post_rerank,
    _question_candidate,
    _select_with_source_priority,
)
from app.llms.models import MODEL_DISTANCE_THRESHOLDS, Model
from app.llms.query_modes import QueryTransformMode
from app.llms.query_variants import (
    build_query_variants,
)
from app.llms.retrieval_budget import retrieval_budget
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
    ann_ideal: bool = False
    ann_prod: bool = False
    final: bool = False
    rank: int | None = None
    n_candidates: int = 0
    best_distance: float | None = None


@dataclass
class Aggregate:
    results: list[Result] = field(default_factory=list)
    model_threshold: float = 0.5

    def _subset(self, pred) -> list[Result]:
        return [r for r in self.results if pred(r)]

    def report(self) -> str:
        retr = self._subset(lambda r: not r.example.is_abstain)
        abst = self._subset(lambda r: r.example.is_abstain)
        lines: list[str] = []

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


async def evaluate_one(
    db: Database,
    ex: Example,
    *,
    embedding_model: Model,
    qt_model: Model,
    initial_k: int,
    top_k: int,
    ideal_limit: int,
    transform_mode: QueryTransformMode,
) -> Result:
    want = anchor_key(ex.anchor)

    variant_bundle = await build_query_variants(ex.query, qt_model, transform_mode)
    variants = [
        (variant.text, variant.is_document) for variant in variant_bundle.variants
    ]
    budget = retrieval_budget(transform_mode, initial_k)
    ideal_probe_limit = max(ideal_limit, budget.per_query_k)

    embeddings = await asyncio.gather(
        *(
            _embed_variant(text, embedding_model, is_document=is_doc)
            for text, is_doc in variants
        ),
    )

    # --- ANN ideal (no threshold) ---
    ideal_lists = await asyncio.gather(
        *(
            asyncio.gather(
                get_closest_questions(
                    db,
                    emb,
                    embedding_model,
                    limit=ideal_probe_limit,
                    threshold=NO_THRESHOLD,
                ),
                get_closest_chunks(
                    db,
                    emb,
                    embedding_model,
                    limit=ideal_probe_limit,
                    threshold=NO_THRESHOLD,
                ),
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

    # --- ANN prod (real per-model threshold) + build rerank candidate pool ---
    prod_lists = await asyncio.gather(
        *(
            asyncio.gather(
                get_closest_questions(
                    db,
                    emb,
                    embedding_model,
                    limit=budget.per_query_k,
                ),
                get_closest_chunks(
                    db,
                    emb,
                    embedding_model,
                    limit=budget.per_query_k,
                ),
            )
            for emb in embeddings
        ),
    )

    seen: set[str] = set()
    cand_keys: list[str] = []
    candidates: list[_Candidate] = []
    cand_rerank: list[str] = []
    cand_nat: list[NaturalKey] = []
    ann_prod = False
    for qs, cs in prod_lists:
        for q in qs:
            cand = _question_candidate(q)
            if cand.key in seen:
                continue
            seen.add(cand.key)
            cand_keys.append(cand.key)
            candidates.append(cand)
            cand_rerank.append(cand.rerank_text)
            cand_nat.append(question_key(q))
            if question_key(q) == want:
                ann_prod = True
        for ch in cs:
            cand = _chunk_candidate(ch)
            if cand.key in seen:
                continue
            seen.add(cand.key)
            cand_keys.append(cand.key)
            candidates.append(cand)
            cand_rerank.append(cand.rerank_text)
            cand_nat.append(chunk_key(ch))
            if chunk_key(ch) == want:
                ann_prod = True

    final_hit = False
    rank: int | None = None
    if cand_rerank:
        response = await _post_rerank(
            {"query": variant_bundle.rerank_query, "documents": cand_rerank},
        )
        ranked = response.json()["reranked_documents"]
        kept_candidates: list[_Candidate] = []
        for item in ranked:
            idx = item["index"]
            if not 0 <= idx < len(cand_nat):
                continue
            if item["score"] < settings.RERANKER_MIN_SCORE:
                continue
            kept_candidates.append(candidates[idx])
        if (
            not kept_candidates and ranked
        ):  # production keeps the single best if all below floor
            best_idx = ranked[0]["index"]
            if 0 <= best_idx < len(cand_nat):
                kept_candidates = [candidates[best_idx]]
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
        best_distance=round(best_distance, 4) if best_distance is not None else None,
    )


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
            ),
        )
    return examples


async def main_async(ns: argparse.Namespace) -> int:
    examples = load_golden(Path(ns.golden))
    if ns.limit:
        examples = examples[: ns.limit]
    embedding_model = Model(ns.embedding_model)
    qt_model = Model(ns.query_transform_model)
    transform_mode = QueryTransformMode.RAW if ns.no_transform else ns.transform_mode
    budget = retrieval_budget(transform_mode, ns.initial_k)
    ideal_probe_limit = max(ns.ideal_limit, budget.per_query_k)

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
                        qt_model=qt_model,
                        initial_k=ns.initial_k,
                        top_k=ns.top_k,
                        ideal_limit=ideal_probe_limit,
                        transform_mode=transform_mode,
                    )
                except Exception as exc:  # one bad example shouldn't abort the run
                    print(
                        f"  ! error on {ex.id}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    return Result(example=ex)

        agg.results = await asyncio.gather(*(run(ex) for ex in examples))
    finally:
        await db.disconnect()
        await close_http_client()

    header = (
        f"golden={ns.golden}  n={len(examples)}  embed={embedding_model.value}  "
        f"qt={qt_model.value}  transform_mode={transform_mode.value}  "
        f"initial_k={budget.initial_k}  per_query_k={budget.per_query_k}  ideal_limit={ideal_probe_limit}  top_k={ns.top_k}  reranker_min={settings.RERANKER_MIN_SCORE}"
    )
    print(header)
    print(agg.report())

    if ns.json:
        out = {
            "config": {
                "embedding_model": embedding_model.value,
                "query_transform_model": qt_model.value,
                "query_transform_mode": transform_mode.value,
                "initial_k": budget.initial_k,
                "per_query_k": budget.per_query_k,
                "ideal_limit": ideal_probe_limit,
                "top_k": ns.top_k,
                "reranker_min_score": settings.RERANKER_MIN_SCORE,
            },
            "results": [
                {
                    "id": r.example.id,
                    "difficulty": r.example.difficulty,
                    "category": r.example.category,
                    "anchor": r.example.anchor,
                    "ann_ideal": r.ann_ideal,
                    "ann_prod": r.ann_prod,
                    "final": r.final,
                    "rank": r.rank,
                    "best_distance": r.best_distance,
                }
                for r in agg.results
            ],
        }
        Path(ns.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote per-example JSON to {ns.json}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FINKI retrieval-evaluation harness")
    p.add_argument("--golden", default=str(Path(__file__).parent / "golden.jsonl"))
    p.add_argument("--embedding-model", default=Model.BGE_M3_LOCAL.value)
    p.add_argument("--query-transform-model", default=Model.GPT_5_4_MINI.value)
    p.add_argument(
        "--transform-mode",
        choices=list(QueryTransformMode),
        default=QueryTransformMode.REWRITE_HYDE,
        type=QueryTransformMode,
        help="query variants to compare: raw, rewrite, hyde, or rewrite_hyde",
    )
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
    p.add_argument(
        "--no-transform",
        action="store_true",
        help="legacy alias for --transform-mode raw",
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
