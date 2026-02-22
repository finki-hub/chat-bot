from collections.abc import AsyncGenerator

from app.llms.qwen_factory import stream_qwen_response

_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"


async def stream_qwen2_1_5b_response(
    user_prompt: str,
    system_prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> AsyncGenerator[str]:
    """
    Streams a response from the Qwen2-1.5B-Instruct model.
    Delegates to the shared Qwen factory.
    """
    async for chunk in stream_qwen_response(
        _MODEL_ID,
        user_prompt,
        system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ):
        yield chunk
