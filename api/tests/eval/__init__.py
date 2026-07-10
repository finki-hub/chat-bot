import argparse
from pathlib import Path
from typing import Final

_EVAL_DIR: Final = Path(__file__).resolve().parent


def eval_json_path(value: str) -> Path:
    try:
        return resolve_eval_json_path(Path(value))
    except (OSError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def resolve_eval_json_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix != ".json":
        raise ValueError(f"{path}: must be an existing .json file")
    return resolved


def eval_jsonl_name(value: str) -> Path:
    path = Path(value)
    if path.name != value or path.suffix != ".jsonl":
        raise argparse.ArgumentTypeError(
            "must be a .jsonl filename without directories",
        )
    try:
        resolved = (_EVAL_DIR / path).resolve(strict=True)
    except OSError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not resolved.is_file() or not resolved.is_relative_to(_EVAL_DIR):
        raise argparse.ArgumentTypeError("must reference a file inside tests/eval")
    return resolved
