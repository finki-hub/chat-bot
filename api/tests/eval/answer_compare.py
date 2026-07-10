import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from . import eval_jsonl_name
from .answer_eval import (
    AnswerCase,
    AnswerEvalError,
    AnswerScore,
    evaluate_results,
    load_answer_cases,
)


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    baseline: AnswerScore
    current: AnswerScore


@dataclass(frozen=True, slots=True)
class AnswerComparison:
    fixed: tuple[AnswerDelta, ...]
    regressions: tuple[AnswerDelta, ...]
    unchanged_failures: tuple[AnswerDelta, ...]


def compare_results(
    cases: tuple[AnswerCase, ...],
    baseline_path: Path,
    current_path: Path,
) -> AnswerComparison:
    baseline = evaluate_results(cases, baseline_path)
    current = evaluate_results(cases, current_path)
    deltas = tuple(
        AnswerDelta(baseline=old, current=new)
        for old, new in zip(baseline, current, strict=True)
    )
    return AnswerComparison(
        fixed=tuple(
            delta
            for delta in deltas
            if not delta.baseline.passed and delta.current.passed
        ),
        regressions=tuple(
            delta
            for delta in deltas
            if delta.baseline.passed and not delta.current.passed
        ),
        unchanged_failures=tuple(
            delta
            for delta in deltas
            if not delta.baseline.passed and not delta.current.passed
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare answer-eval result sets.")
    parser.add_argument("--baseline", type=eval_jsonl_name, required=True)
    parser.add_argument("--current", type=eval_jsonl_name, required=True)
    args = parser.parse_args()
    try:
        comparison = compare_results(
            load_answer_cases(Path(__file__).with_name("answer_golden.jsonl")),
            args.baseline,
            args.current,
        )
    except AnswerEvalError as exc:
        print(exc, file=sys.stderr)
        return 2
    for label, deltas in (
        ("FIXED", comparison.fixed),
        ("REGRESSION", comparison.regressions),
        ("UNCHANGED-FAIL", comparison.unchanged_failures),
    ):
        for delta in deltas:
            print(f"{label}: {delta.current.case_id}")
    return int(bool(comparison.regressions))


if __name__ == "__main__":
    raise SystemExit(main())
