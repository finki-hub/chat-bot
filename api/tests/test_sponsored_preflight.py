import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = REPO_ROOT / "scripts" / "verify-sponsored-model-preflight.sh"


@pytest.mark.parametrize(
    ("stream", "expected_success"),
    [
        (
            'event: error\ndata: {"code":"agent_error"}\n\nevent: done\ndata: {}\n\n',
            False,
        ),
        ('event: token\ndata: {"text":"ok"}\n\nevent: done\ndata: {}\n\n', True),
    ],
)
def test_sponsored_preflight_rejects_error_streams(
    tmp_path: Path,
    stream: str,
    expected_success: bool,
) -> None:
    stream_path = tmp_path / "preflight.sse"
    stream_path.write_text(stream)

    result = subprocess.run(  # noqa: S603
        ["/bin/sh", str(VERIFY_SCRIPT), str(stream_path)],
        check=False,
    )

    assert (result.returncode == 0) is expected_success
