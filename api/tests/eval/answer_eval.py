import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

_URL_PATTERN: Final = re.compile(r"https?://[^\s,)>\]]+")
_REFUSAL_MARKERS: Final = (
    "можам да помогнам само",
    "немам доволно",
    "нема доволно",
    "не можев да пронајдам",
    "не можам да одговорам",
)
_GOLDEN_CASES: Final = Path(__file__).with_name("answer_golden.jsonl")
_RESULTS_PATH: Final = Path(__file__).with_name("answer_results.jsonl")


@dataclass(frozen=True, slots=True)
class AnswerEvalError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class AnswerExpectation:
    required_sources: tuple[str, ...]
    forbidden: tuple[str, ...]
    max_urls: int
    min_cyrillic_ratio: float
    requires_refusal: bool


@dataclass(frozen=True, slots=True)
class AnswerCase:
    id: str
    category: str
    expectation: AnswerExpectation


@dataclass(frozen=True, slots=True)
class AnswerScore:
    case_id: str
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def _mapping(value: JsonValue, path: str) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return value
    raise AnswerEvalError(f"{path}: expected object")


def _text(value: JsonValue, path: str) -> str:
    if isinstance(value, str):
        return value
    raise AnswerEvalError(f"{path}: expected string")


def _integer(value: JsonValue, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnswerEvalError(f"{path}: expected integer")
    return value


def _number(value: JsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AnswerEvalError(f"{path}: expected number")
    return float(value)


def _flag(value: JsonValue, path: str) -> bool:
    if isinstance(value, bool):
        return value
    raise AnswerEvalError(f"{path}: expected boolean")


def _texts(value: JsonValue, path: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AnswerEvalError(f"{path}: expected array")
    return tuple(_text(item, f"{path}[{index}]") for index, item in enumerate(value))


def _expectation(value: JsonValue, path: str) -> AnswerExpectation:
    row = _mapping(value, path)
    max_urls = _integer(row.get("max_urls"), f"{path}.max_urls")
    min_cyrillic_ratio = _number(
        row.get("min_cyrillic_ratio"),
        f"{path}.min_cyrillic_ratio",
    )
    if max_urls < 0:
        raise AnswerEvalError(f"{path}.max_urls: must be non-negative")
    if not 0 <= min_cyrillic_ratio <= 1:
        raise AnswerEvalError(f"{path}.min_cyrillic_ratio: must be between 0 and 1")
    return AnswerExpectation(
        required_sources=_texts(
            row.get("required_sources"),
            f"{path}.required_sources",
        ),
        forbidden=_texts(row.get("forbidden"), f"{path}.forbidden"),
        max_urls=max_urls,
        min_cyrillic_ratio=min_cyrillic_ratio,
        requires_refusal=_flag(
            row.get("requires_refusal"),
            f"{path}.requires_refusal",
        ),
    )


def _case(value: JsonValue, path: str) -> AnswerCase:
    row = _mapping(value, path)
    return AnswerCase(
        id=_text(row.get("id"), f"{path}.id"),
        category=_text(row.get("category"), f"{path}.category"),
        expectation=_expectation(row.get("expectation"), f"{path}.expectation"),
    )


def load_answer_cases(path: Path) -> tuple[AnswerCase, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnswerEvalError(f"{path}: {exc}") from exc

    cases: list[AnswerCase] = []
    ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value: JsonValue = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnswerEvalError(f"{path}: line {line_number}: {exc.msg}") from exc
        case = _case(value, f"{path}: line {line_number}")
        if case.id in ids:
            raise AnswerEvalError(f"{path}: duplicate id {case.id}")
        ids.add(case.id)
        cases.append(case)
    return tuple(cases)


def _cyrillic_ratio(answer: str) -> float:
    letters = [character for character in answer if character.isalpha()]
    if not letters:
        return 0.0
    cyrillic = sum("\u0400" <= character <= "\u04ff" for character in letters)
    return cyrillic / len(letters)


def score_answer(case: AnswerCase, answer: str) -> AnswerScore:
    folded = answer.casefold()
    failures = [
        f"missing-source:{source}"
        for source in case.expectation.required_sources
        if source.casefold() not in folded
    ]
    failures.extend(
        f"forbidden:{phrase}"
        for phrase in case.expectation.forbidden
        if phrase.casefold() in folded
    )
    url_count = len(_URL_PATTERN.findall(answer))
    if url_count > case.expectation.max_urls:
        failures.append(f"too-many-urls:{url_count}>{case.expectation.max_urls}")
    ratio = _cyrillic_ratio(answer)
    if ratio < case.expectation.min_cyrillic_ratio:
        failures.append(
            f"low-cyrillic-ratio:{ratio:.2f}<{case.expectation.min_cyrillic_ratio:.2f}",
        )
    if case.expectation.requires_refusal and not any(
        marker in folded for marker in _REFUSAL_MARKERS
    ):
        failures.append("missing-refusal")
    return AnswerScore(case_id=case.id, failures=tuple(failures))


def _load_results(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AnswerEvalError(f"{path}: {exc}") from exc

    results: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value: JsonValue = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnswerEvalError(f"{path}: line {line_number}: {exc.msg}") from exc
        row = _mapping(value, f"{path}: line {line_number}")
        case_id = _text(row.get("id"), f"{path}: line {line_number}.id")
        if case_id in results:
            raise AnswerEvalError(f"{path}: duplicate id {case_id}")
        results[case_id] = _text(
            row.get("answer"),
            f"{path}: line {line_number}.answer",
        )
    return results


def evaluate_results(
    cases: tuple[AnswerCase, ...],
    results_path: Path,
) -> tuple[AnswerScore, ...]:
    results = _load_results(results_path)
    known_ids = {case.id for case in cases}
    unknown_ids = results.keys() - known_ids
    if unknown_ids:
        raise AnswerEvalError(
            f"{results_path}: unknown ids {', '.join(sorted(unknown_ids))}",
        )
    return tuple(
        score_answer(case, results[case.id])
        if case.id in results
        else AnswerScore(case_id=case.id, failures=("missing-answer",))
        for case in cases
    )


def main() -> int:
    try:
        scores = evaluate_results(load_answer_cases(_GOLDEN_CASES), _RESULTS_PATH)
    except AnswerEvalError as exc:
        print(exc, file=sys.stderr)
        return 2
    for score in scores:
        status = "PASS" if score.passed else f"FAIL ({', '.join(score.failures)})"
        print(f"{score.case_id}: {status}")
    return int(any(not score.passed for score in scores))


if __name__ == "__main__":
    raise SystemExit(main())
