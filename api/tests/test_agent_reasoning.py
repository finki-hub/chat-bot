from langchain_core.messages import AIMessageChunk

from app.llms.agents import _chunk_reasoning


def test_chunk_reasoning_extracts_openrouter_text_details() -> None:
    chunk = AIMessageChunk(
        content="",
        additional_kwargs={
            "reasoning_details": [
                {"type": "reasoning.text", "text": "first"},
                {"type": "reasoning.encrypted", "data": "opaque"},
                {"type": "reasoning.text", "text": " second"},
            ],
        },
    )

    assert _chunk_reasoning(chunk) == "first second"
