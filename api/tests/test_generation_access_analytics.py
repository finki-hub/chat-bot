import asyncio

import pytest

from app.api import chat as chat_api
from app.llms.models import Model
from tests.chat_models_access_support import credentials, settings
from tests.test_chat_sponsored_admission import _admission, _run_stream


@pytest.mark.parametrize("byok", [False, True])
def test_generation_access_mode_follows_resolved_credential_path(monkeypatch, byok):
    events = []
    monkeypatch.setattr(
        chat_api,
        "capture",
        lambda distinct_id, event, properties: events.append((event, properties)),
    )

    async def admit(db, *, request_id, **kwargs):
        return _admission(request_id)

    asyncio.run(
        _run_stream(
            monkeypatch,
            current_settings=settings(enabled=True),
            user_credentials=credentials(openai=byok),
            admit=admit,
            inference_model=Model.GPT_5_6_LUNA,
        )
    )
    generation = next(props for event, props in events if event == "$ai_generation")
    assert generation["model_access_mode"] == ("byok" if byok else "sponsored")
