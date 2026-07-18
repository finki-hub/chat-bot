import logging

import anyio
from anyio.lowlevel import checkpoint

from app.llms import agents as agents_module

_PRIVATE_PROVIDER_ERROR = (
    "provider rejected key=secret at https://private-provider.example/v1"
)


class _FailingAgent:
    async def astream_events(self, *args, **kwargs):
        del args, kwargs
        await checkpoint()
        yield {"event": "on_chain_start"}
        raise RuntimeError(_PRIVATE_PROVIDER_ERROR)


def test_agent_stream_failure_omits_raw_error(caplog):
    caplog.set_level(logging.ERROR, logger=agents_module.__name__)

    async def collect():
        return [
            chunk
            async for chunk in agents_module.create_agent_token_generator(
                _FailingAgent(),
                [],
            )
        ]

    chunks = anyio.run(collect)

    assert _PRIVATE_PROVIDER_ERROR not in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert any('"code": "agent_error"' in chunk for chunk in chunks)
