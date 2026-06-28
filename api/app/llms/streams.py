import logging

from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage

from app.llms.agents import StreamObservation
from app.llms.anthropic import stream_anthropic_agent_response
from app.llms.google import stream_google_agent_response
from app.llms.gpu_api import stream_gpu_api_response
from app.llms.models import REASONING_CAPABLE_MODELS, Model
from app.llms.ollama import stream_ollama_agent_response
from app.llms.openai import stream_openai_agent_response

logger = logging.getLogger(__name__)


def _tag_provider(observation: StreamObservation | None, provider: str) -> None:
    if observation is not None:
        observation.provider = provider


async def stream_response_with_agent(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    history: list[BaseMessage],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
    observation: StreamObservation | None = None,
) -> StreamingResponse:
    """
    Stream a response from the specified model using the provided user prompt and system prompt with agent.
    """
    logger.info(
        "Streaming response with agent for user prompt length: '%d' "
        "with model: %s and %d prior turn(s)",
        len(user_prompt),
        model.value,
        len(history),
    )

    reasoning = reasoning and model in REASONING_CAPABLE_MODELS

    if observation is not None:
        observation.model = model.value

    match model:
        case (
            Model.LLAMA_3_3_70B
            | Model.MISTRAL
            | Model.DEEPSEEK_R1_70B
            | Model.QWEN2_5_72B
            | Model.DOMESTIC_YAK_8B_INSTRUCT_GGUF
            | Model.VEZILKALLM_GGUF
        ):
            _tag_provider(observation, "ollama")
            return await stream_ollama_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                observation=observation,
            )

        case (
            Model.GPT_4O_MINI
            | Model.GPT_4_1
            | Model.GPT_4_1_MINI
            | Model.GPT_4_1_NANO
            | Model.GPT_5_4
            | Model.GPT_5_4_MINI
            | Model.GPT_5_2
            | Model.GPT_5_MINI
            | Model.GPT_5_NANO
        ):
            _tag_provider(observation, "openai")
            return await stream_openai_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                observation=observation,
            )

        case (
            Model.GEMINI_2_5_FLASH | Model.GEMINI_2_5_PRO | Model.GEMINI_3_FLASH_PREVIEW
        ):
            _tag_provider(observation, "google")
            return await stream_google_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                observation=observation,
            )

        case (
            Model.CLAUDE_OPUS_4_8
            | Model.CLAUDE_OPUS_4_7
            | Model.CLAUDE_SONNET_4_6
            | Model.CLAUDE_HAIKU_4_5
        ):
            _tag_provider(observation, "anthropic")
            return await stream_anthropic_agent_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning=reasoning,
                observation=observation,
            )

        case Model.QWEN2_1_5_B_INSTRUCT | Model.QWEN2_5_7B_INSTRUCT:
            _tag_provider(observation, "gpu_api")
            return stream_gpu_api_response(
                user_prompt,
                model,
                system_prompt=system_prompt,
                history=history,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                observation=observation,
            )

        case _:
            raise ValueError(f"Unsupported model: {model}")
