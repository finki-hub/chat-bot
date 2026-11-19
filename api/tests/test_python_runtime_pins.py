import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PYTHON_315 = re.compile(r"3\.15(?:\.\d+)?(?:-|$)")


def _assert_image_python_315(
    dockerfile: str,
    image: str,
    stage: str,
) -> None:
    prefix = f"FROM {image}:"
    suffix = f" AS {stage}"
    line = next(
        (
            line
            for line in dockerfile.splitlines()
            if line.startswith(prefix) and line.endswith(suffix)
        ),
        None,
    )
    assert line is not None, f"missing {image} {stage} stage"

    tag = line.removeprefix(prefix).removesuffix(suffix)
    assert PYTHON_315.search(tag), f"{image}:{tag} is not a Python 3.15 image"
    assert "3.14" not in tag, f"{image}:{tag} is a Python 3.14 image"


def test_python_runtime_is_315_everywhere() -> None:
    api_dockerfile = (REPO_ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    gpu_dockerfile = (REPO_ROOT / "gpu-api" / "Dockerfile").read_text(
        encoding="utf-8",
    )

    _assert_image_python_315(
        api_dockerfile,
        "ghcr.io/astral-sh/uv",
        "builder",
    )
    _assert_image_python_315(api_dockerfile, "python", "final")
    _assert_image_python_315(
        gpu_dockerfile,
        "ghcr.io/astral-sh/uv",
        "builder",
    )
    _assert_image_python_315(gpu_dockerfile, "python", "final")

    for package in ("api", "gpu-api"):
        version = (REPO_ROOT / package / ".python-version").read_text(
            encoding="utf-8",
        )
        assert version.strip() == "3.15"

        pyproject = tomllib.loads(
            (REPO_ROOT / package / "pyproject.toml").read_text(encoding="utf-8"),
        )
        assert pyproject["project"]["requires-python"] == ">=3.15,<3.16"
