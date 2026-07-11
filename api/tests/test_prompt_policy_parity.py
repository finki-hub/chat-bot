from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "filename",
    ["agent_system.txt", "discord_format.txt", "web_format.txt"],
)
def test_api_and_gpu_prompt_policies_match(filename: str):
    api_prompt = (_ROOT / "api" / "resources" / "prompts" / filename).read_text(
        encoding="utf-8",
    )
    gpu_prompt = (_ROOT / "gpu-api" / "resources" / "prompts" / filename).read_text(
        encoding="utf-8",
    )

    assert api_prompt == gpu_prompt
