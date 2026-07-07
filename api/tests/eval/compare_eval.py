import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from . import eval_json_path, resolve_eval_json_path

AnchorType = Literal["Q", "C", "none"]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

BUCKETS: Final = (
    "overall",
    "source=faq",
    "source=chunk",
    "difficulty=easy",
    "difficulty=hard",
    "abstain",
)


class EvalJsonError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    anchor_type: AnchorType
    difficulty: str
    category: str
    ann_ideal: bool
    ann_prod: bool
    final: bool
    rank: int | None

    @property
    def is_abstain(self) -> bool:
        return self.anchor_type == "none"

    @property
    def succeeded(self) -> bool:
        return not self.final if self.is_abstain else self.final

    @property
    def failure_reason(self) -> str:
        if self.succeeded:
            return "PASS"
        if self.is_abstain:
            return "ABSTAIN-LEAK"
        if not self.ann_ideal:
            return "ANN-MISS"
        if not self.ann_prod:
            return "ANN-PROD-MISS"
        return "RERANK-MISS"


@dataclass(frozen=True, slots=True)
class BucketSummary:
    count: int
    final_count: int
    mrr: float

    @property
    def final_rate(self) -> float:
        return 0.0 if self.count == 0 else self.final_count / self.count


@dataclass(frozen=True, slots=True)
class CaseDelta:
    baseline: EvalCase
    current: EvalCase


@dataclass(frozen=True, slots=True)
class EvalComparison:
    bucket_deltas: dict[str, tuple[BucketSummary, BucketSummary]]
    fixed: list[CaseDelta]
    new_regressions: list[CaseDelta]
    unchanged_misses: list[CaseDelta]


def _mapping(value: JsonValue, path: str) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    raise EvalJsonError(f"{path}: expected object")


def _items(value: JsonValue, path: str) -> list[JsonValue]:
    if isinstance(value, list):
        return value
    raise EvalJsonError(f"{path}: expected array")


def _text(value: JsonValue, path: str) -> str:
    if isinstance(value, str):
        return value
    raise EvalJsonError(f"{path}: expected string")


def _flag(value: JsonValue, path: str) -> bool:
    if isinstance(value, bool):
        return value
    raise EvalJsonError(f"{path}: expected boolean")


def _rank(value: JsonValue, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise EvalJsonError(f"{path}: expected integer or null")


def _anchor_type(value: JsonValue, path: str) -> AnchorType:
    raw = _text(_mapping(value, path).get("type"), f"{path}.type")
    if raw == "Q":
        return "Q"
    if raw == "C":
        return "C"
    if raw == "none":
        return "none"
    raise EvalJsonError(f"{path}.type: expected Q, C, or none")


def _case(value: JsonValue, path: str) -> EvalCase:
    row = _mapping(value, path)
    return EvalCase(
        id=_text(row.get("id"), f"{path}.id"),
        anchor_type=_anchor_type(row.get("anchor"), f"{path}.anchor"),
        difficulty=_text(row.get("difficulty", ""), f"{path}.difficulty"),
        category=_text(row.get("category", ""), f"{path}.category"),
        ann_ideal=_flag(row.get("ann_ideal"), f"{path}.ann_ideal"),
        ann_prod=_flag(row.get("ann_prod"), f"{path}.ann_prod"),
        final=_flag(row.get("final"), f"{path}.final"),
        rank=_rank(row.get("rank"), f"{path}.rank"),
    )


def _cases(data: dict[str, JsonValue], path: str) -> dict[str, EvalCase]:
    rows = _items(data.get("results"), f"{path}.results")
    parsed = [_case(row, f"{path}.results[{index}]") for index, row in enumerate(rows)]
    return {case.id: case for case in parsed}


def load_eval(path: Path) -> dict[str, EvalCase]:
    try:
        safe_path = resolve_eval_json_path(path)
        data: JsonValue = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise EvalJsonError(f"{path}: {exc}") from exc
    return _cases(_mapping(data, str(safe_path)), str(safe_path))


def _in_bucket(case: EvalCase, bucket: str) -> bool:
    if bucket == "overall":
        return not case.is_abstain
    if bucket == "source=faq":
        return case.anchor_type == "Q"
    if bucket == "source=chunk":
        return case.anchor_type == "C"
    if bucket == "difficulty=easy":
        return (not case.is_abstain) and case.difficulty == "easy"
    if bucket == "difficulty=hard":
        return (not case.is_abstain) and case.difficulty == "hard"
    if bucket == "abstain":
        return case.is_abstain
    raise EvalJsonError(f"unknown bucket: {bucket}")


def _summary(cases: list[EvalCase], bucket: str) -> BucketSummary:
    bucket_cases = [case for case in cases if _in_bucket(case, bucket)]
    final_count = sum(1 for case in bucket_cases if case.final)
    mrr = (
        sum(1 / case.rank for case in bucket_cases if case.rank is not None)
        / len(bucket_cases)
        if bucket_cases
        else 0.0
    )
    return BucketSummary(len(bucket_cases), final_count, mrr)


def compare_runs(
    baseline: dict[str, JsonValue],
    current: dict[str, JsonValue],
) -> EvalComparison:
    return compare_cases(_cases(baseline, "baseline"), _cases(current, "current"))


def compare_cases(
    baseline: dict[str, EvalCase],
    current: dict[str, EvalCase],
) -> EvalComparison:
    baseline_ids = set(baseline)
    current_ids = set(current)
    if baseline_ids != current_ids:
        raise EvalJsonError(
            f"case id mismatch: baseline-only={', '.join(sorted(baseline_ids - current_ids)) or 'none'}; current-only={', '.join(sorted(current_ids - baseline_ids)) or 'none'}",
        )
    pairs = [
        (baseline[id_], current[id_]) for id_ in sorted(baseline_ids & current_ids)
    ]
    baseline_cases = [baseline_case for baseline_case, _current_case in pairs]
    current_cases = [current_case for _baseline_case, current_case in pairs]
    bucket_deltas = {
        bucket: (_summary(baseline_cases, bucket), _summary(current_cases, bucket))
        for bucket in BUCKETS
    }
    fixed = [
        CaseDelta(base, cur)
        for base, cur in pairs
        if not base.succeeded and cur.succeeded
    ]
    regressions = [
        CaseDelta(base, cur)
        for base, cur in pairs
        if base.succeeded and not cur.succeeded
    ]
    unchanged = [
        CaseDelta(base, cur)
        for base, cur in pairs
        if not base.succeeded and not cur.succeeded
    ]
    return EvalComparison(
        bucket_deltas=bucket_deltas,
        fixed=fixed,
        new_regressions=regressions,
        unchanged_misses=unchanged,
    )


def _format_rate(summary: BucketSummary) -> str:
    return (
        "n=0"
        if summary.count == 0
        else f"{summary.final_count}/{summary.count} ({100 * summary.final_rate:.1f}%)"
    )


def _append_cases(lines: list[str], title: str, cases: list[CaseDelta]) -> None:
    lines.append(title)
    lines.extend(
        [
            f"  {case.current.id} ({case.current.difficulty}/{case.current.category}) {case.baseline.failure_reason} -> {case.current.failure_reason}"
            for case in cases
        ]
        if cases
        else ["  none"],
    )


def render_report(comparison: EvalComparison, *, max_regressions: int = 0) -> str:
    lines = ["Retrieval eval comparison", "", "Buckets"]
    for bucket, (baseline, current) in comparison.bucket_deltas.items():
        delta = 100 * (current.final_rate - baseline.final_rate)
        lines.append(
            f"  {bucket}: {_format_rate(baseline)} -> {_format_rate(current)} "
            f"({delta:+.1f} pp, MRR {baseline.mrr:.3f} -> {current.mrr:.3f})",
        )
    lines.append("")
    _append_cases(lines, "Fixed cases", comparison.fixed)
    lines.append("")
    _append_cases(lines, "New regressions", comparison.new_regressions)
    lines.append("")
    _append_cases(lines, "Unchanged misses", comparison.unchanged_misses)
    decision = "FAIL" if len(comparison.new_regressions) > max_regressions else "PASS"
    lines.append("")
    lines.append(
        f"Decision: {decision} ({len(comparison.new_regressions)} new regressions, budget {max_regressions})",
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval eval JSON outputs")
    parser.add_argument("--baseline", required=True, type=eval_json_path)
    parser.add_argument("--current", required=True, type=eval_json_path)
    parser.add_argument("--max-regressions", default=0, type=_non_negative_int)
    ns = parser.parse_args(argv)
    try:
        comparison = compare_cases(load_eval(ns.baseline), load_eval(ns.current))
    except EvalJsonError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(render_report(comparison, max_regressions=ns.max_regressions))
    return 1 if len(comparison.new_regressions) > ns.max_regressions else 0


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
