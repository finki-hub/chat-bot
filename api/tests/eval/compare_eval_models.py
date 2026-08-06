import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_budget import retrieval_budget

from . import resolve_eval_json_path

AnchorType = Literal["Q", "C", "none"]
TransformMode = Literal["raw", "rewrite", "hyde", "rewrite_hyde"]
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class EvalJsonError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class EffectiveRetrievalConfig:
    transform_mode: TransformMode
    initial_k: int
    per_query_k: int


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
    requested_config: EffectiveRetrievalConfig
    effective_config: EffectiveRetrievalConfig | None

    @property
    def is_abstain(self) -> bool:
        return self.anchor_type == "none"

    @property
    def succeeded(self) -> bool:
        return not self.final if self.is_abstain else self.final

    @property
    def used_requested_config(self) -> bool:
        return self.effective_config == self.requested_config

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


def _optional_int(value: JsonValue, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise EvalJsonError(f"{path}: expected integer or null")


def _required_int(value: JsonValue, path: str) -> int:
    parsed = _optional_int(value, path)
    if parsed is None:
        raise EvalJsonError(f"{path}: expected integer")
    return parsed


def _transform_mode(value: JsonValue, path: str) -> TransformMode:
    raw = _text(value, path)
    if raw == "raw":
        return "raw"
    if raw == "rewrite":
        return "rewrite"
    if raw == "hyde":
        return "hyde"
    if raw == "rewrite_hyde":
        return "rewrite_hyde"
    raise EvalJsonError(f"{path}: expected raw, rewrite, hyde, or rewrite_hyde")


def _anchor_type(value: JsonValue, path: str) -> AnchorType:
    raw = _text(_mapping(value, path).get("type"), f"{path}.type")
    if raw == "Q":
        return "Q"
    if raw == "C":
        return "C"
    if raw == "none":
        return "none"
    raise EvalJsonError(f"{path}.type: expected Q, C, or none")


def _effective_config(
    row: dict[str, JsonValue],
    path: str,
    requested_config: EffectiveRetrievalConfig,
) -> EffectiveRetrievalConfig | None:
    keys = (
        "effective_transform_mode",
        "effective_initial_k",
        "effective_per_query_k",
    )
    present = [key in row for key in keys]
    if not any(present):
        return requested_config
    if not all(present):
        raise EvalJsonError(f"{path}: effective retrieval configuration is partial")
    values = [row[key] for key in keys]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise EvalJsonError(f"{path}: effective retrieval configuration is partial")
    return EffectiveRetrievalConfig(
        transform_mode=_transform_mode(
            values[0],
            f"{path}.effective_transform_mode",
        ),
        initial_k=_required_int(values[1], f"{path}.effective_initial_k"),
        per_query_k=_required_int(values[2], f"{path}.effective_per_query_k"),
    )


def _case(
    value: JsonValue,
    path: str,
    requested_config: EffectiveRetrievalConfig,
) -> EvalCase:
    row = _mapping(value, path)
    return EvalCase(
        id=_text(row.get("id"), f"{path}.id"),
        anchor_type=_anchor_type(row.get("anchor"), f"{path}.anchor"),
        difficulty=_text(row.get("difficulty", ""), f"{path}.difficulty"),
        category=_text(row.get("category", ""), f"{path}.category"),
        ann_ideal=_flag(row.get("ann_ideal"), f"{path}.ann_ideal"),
        ann_prod=_flag(row.get("ann_prod"), f"{path}.ann_prod"),
        final=_flag(row.get("final"), f"{path}.final"),
        rank=_optional_int(row.get("rank"), f"{path}.rank"),
        requested_config=requested_config,
        effective_config=_effective_config(row, path, requested_config),
    )


def _requested_config(
    config: dict[str, JsonValue],
    path: str,
) -> EffectiveRetrievalConfig:
    nominal_keys = ("query_transform_mode", "initial_k", "per_query_k")
    nominal_present = [key in config for key in nominal_keys]
    if all(nominal_present):
        return EffectiveRetrievalConfig(
            transform_mode=_transform_mode(
                config["query_transform_mode"],
                f"{path}.query_transform_mode",
            ),
            initial_k=_required_int(config["initial_k"], f"{path}.initial_k"),
            per_query_k=_required_int(
                config["per_query_k"],
                f"{path}.per_query_k",
            ),
        )
    if any(nominal_present):
        raise EvalJsonError(f"{path}: requested retrieval configuration is partial")

    prior_keys = ("requested_query_transform_mode", "requested_initial_k")
    prior_present = [key in config for key in prior_keys]
    if all(prior_present):
        transform_mode = _transform_mode(
            config["requested_query_transform_mode"],
            f"{path}.requested_query_transform_mode",
        )
        requested_initial_k = _required_int(
            config["requested_initial_k"],
            f"{path}.requested_initial_k",
        )
        budget = retrieval_budget(
            QueryTransformMode(transform_mode),
            requested_initial_k,
        )
        return EffectiveRetrievalConfig(
            transform_mode=transform_mode,
            initial_k=budget.initial_k,
            per_query_k=budget.per_query_k,
        )
    if any(prior_present):
        raise EvalJsonError(f"{path}: requested retrieval configuration is partial")
    raise EvalJsonError(f"{path}: requested retrieval configuration is missing")


def parse_cases(data: dict[str, JsonValue], path: str) -> dict[str, EvalCase]:
    config_path = f"{path}.config"
    config = _mapping(data.get("config"), config_path)
    requested_config = _requested_config(config, config_path)
    rows = _items(data.get("results"), f"{path}.results")
    parsed: dict[str, EvalCase] = {}
    for index, row in enumerate(rows):
        case = _case(row, f"{path}.results[{index}]", requested_config)
        if case.id in parsed:
            raise EvalJsonError(
                f"{path}.results[{index}]: duplicate case id: {case.id}",
            )
        parsed[case.id] = case
    return parsed


def load_eval(path: Path) -> dict[str, EvalCase]:
    try:
        safe_path = resolve_eval_json_path(path)
        data: JsonValue = json.loads(safe_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise EvalJsonError(f"{path}: {exc}") from exc
    return parse_cases(_mapping(data, str(safe_path)), str(safe_path))
