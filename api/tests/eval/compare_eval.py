import argparse
import sys
from dataclasses import dataclass
from typing import Final

from . import eval_json_path
from .compare_eval_models import (
    EffectiveRetrievalConfig,
    EvalCase,
    EvalJsonError,
    JsonValue,
    load_eval,
    parse_cases,
)

BUCKETS: Final = (
    "overall",
    "source=faq",
    "source=chunk",
    "difficulty=easy",
    "difficulty=hard",
    "abstain",
)


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
    incomparable_configs: list[CaseDelta]


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
    return compare_cases(
        parse_cases(baseline, "baseline"),
        parse_cases(current, "current"),
    )


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
    incomparable_configs = [
        CaseDelta(base, cur)
        for base, cur in pairs
        if not base.used_requested_config or not cur.used_requested_config
    ]
    comparable_pairs = [
        (base, cur)
        for base, cur in pairs
        if base.used_requested_config and cur.used_requested_config
    ]
    baseline_cases = [
        baseline_case for baseline_case, _current_case in comparable_pairs
    ]
    current_cases = [current_case for _baseline_case, current_case in comparable_pairs]
    bucket_deltas = {
        bucket: (_summary(baseline_cases, bucket), _summary(current_cases, bucket))
        for bucket in BUCKETS
    }
    fixed = [
        CaseDelta(base, cur)
        for base, cur in comparable_pairs
        if not base.succeeded and cur.succeeded
    ]
    regressions = [
        CaseDelta(base, cur)
        for base, cur in comparable_pairs
        if base.succeeded and not cur.succeeded
    ]
    unchanged = [
        CaseDelta(base, cur)
        for base, cur in comparable_pairs
        if not base.succeeded and not cur.succeeded
    ]
    return EvalComparison(
        bucket_deltas=bucket_deltas,
        fixed=fixed,
        new_regressions=regressions,
        unchanged_misses=unchanged,
        incomparable_configs=incomparable_configs,
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


def _format_config(config: EffectiveRetrievalConfig | None) -> str:
    if config is None:
        return "unavailable"
    return (
        f"{config.transform_mode}(initial_k={config.initial_k}, "
        f"per_query_k={config.per_query_k})"
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
    lines.append("")
    lines.append("Incomparable effective retrieval configurations")
    lines.extend(
        [
            f"  {case.current.id}: baseline requested="
            f"{_format_config(case.baseline.requested_config)} effective="
            f"{_format_config(case.baseline.effective_config)}; current requested="
            f"{_format_config(case.current.requested_config)} effective="
            f"{_format_config(case.current.effective_config)}"
            for case in comparison.incomparable_configs
        ]
        if comparison.incomparable_configs
        else ["  none"],
    )
    if comparison.incomparable_configs:
        decision = "INVALID"
    elif len(comparison.new_regressions) > max_regressions:
        decision = "FAIL"
    else:
        decision = "PASS"
    lines.append("")
    lines.append(
        f"Decision: {decision} ({len(comparison.new_regressions)} new regressions, "
        f"{len(comparison.incomparable_configs)} incomparable configurations, "
        f"budget {max_regressions})",
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
    if comparison.incomparable_configs:
        return 2
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
