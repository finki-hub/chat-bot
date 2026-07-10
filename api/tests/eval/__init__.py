import argparse
from pathlib import Path


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
