import logging
from collections.abc import Generator
from typing import Final, TypedDict

from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from app.llms.agents import (
    StreamObservation,
    capture_model_fallback,
    content_to_text,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
)
from app.llms.models import OPENROUTER_MANDATORY_REASONING_MODELS, Model
from app.llms.prompts import build_agent_messages
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import get_agent_tools
from app.schemas.chat_credentials import ChatCredentialSecret

logger = logging.getLogger(__name__)

_MODEL_PREFIX = "openrouter:"
_PROVIDER_ROUTING = {
    "allow_fallbacks": True,
    "data_collection": "deny",
    "require_parameters": True,
}
_REASONING_EFFORT_BY_MODEL: Final[dict[Model, str]] = {
    Model.OPENROUTER_DEEPSEEK_V4_PRO_0813: "high",
    Model.OPENROUTER_DEEPSEEK_V4_FLASH_0731: "high",
    Model.OPENROUTER_GLM_5_3: "high",
    Model.OPENROUTER_KIMI_K3: "high",
    Model.OPENROUTER_QWEN3_8_MAX: "high",
    Model.OPENROUTER_QWEN3_8_27B: "medium",
    Model.OPENROUTER_GROK_4_6: "high",
    Model.OPENROUTER_HY3: "high",
}


class _ReasoningConfig(TypedDict, total=False):
    effort: str


def _reasoning_config(model: Model, requested: bool) -> _ReasoningConfig:
    enabled = requested or model in OPENROUTER_MANDATORY_REASONING_MODELS
    if not enabled:
        return {"effort": "none"}
    effort = _REASONING_EFFORT_BY_MODEL.get(model)
    return {} if effort is None else {"effort": effort}


def get_openrouter_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
) -> ChatOpenRouter:
    credential = require_provider_credential("openrouter", credential)
    return ChatOpenRouter.model_validate(
        {
            "model_name": (
                model.value.removeprefix(_MODEL_PREFIX)
                if upstream_model is None
                else upstream_model
            ),
            "openrouter_api_key": SecretStr(credential.api_key),
            "base_url": credential.base_url,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "streaming": True,
            "stream_usage": True,
            "reasoning": _reasoning_config(model, reasoning),
            "openrouter_provider": _PROVIDER_ROUTING,
        },
    )


def stream_openrouter_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    history: list[BaseMessage] | None = None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
) -> StreamingResponse:
    llm = get_openrouter_llm(
        model,
        temperature,
        top_p,
        max_tokens,
        reasoning=reasoning,
        credential=credential,
        upstream_model=upstream_model,
    )
    messages = build_agent_messages(system_prompt, history or [], user_prompt)

    def sync_chunk_gen() -> Generator[AIMessageChunk]:
        yield from llm.stream(messages)

    return stream_sync_gen_as_sse(sync_chunk_gen())


async def stream_openrouter_agent_response(
    user_prompt: str,
    model: Model,
    *,
    system_prompt: str,
    history: list[BaseMessage] | None = None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: bool = False,
    observation: StreamObservation | None = None,
    credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
) -> StreamingResponse:
    try:
        llm = get_openrouter_llm(
            model,
            temperature,
            top_p,
            max_tokens,
            reasoning=reasoning,
            credential=credential,
            upstream_model=upstream_model,
        )
        tools = await get_agent_tools()
        agent = create_agent(llm, tools)
        messages = build_agent_messages(system_prompt, history or [], user_prompt)
        return StreamingResponse(
            create_agent_token_generator(agent, messages, observation),
            media_type="text/event-stream",
        )
    # ruff: ignore[BLE001] -- agent setup falls back to the regular provider stream
    except Exception as exc:
        logger.warning(
            "OpenRouter agent setup failed; using regular response model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )
        capture_model_fallback(
            observation,
            from_model=model.value,
            to_model=model.value,
            reason="agent_setup_failed",
        )
        return stream_openrouter_response(
            user_prompt,
            model,
            system_prompt=system_prompt,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            reasoning=reasoning,
            credential=credential,
            upstream_model=upstream_model,
        )


async def transform_query_with_openrouter(
    query: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    credential: ChatCredentialSecret | None = None,
) -> str:
    try:
        llm = get_openrouter_llm(
            model,
            temperature,
            top_p,
            max_tokens,
            credential=credential,
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=query)],
        )
        return content_to_text(response.content).strip()
    # ruff: ignore[BLE001] -- query transformation is an optional upstream fallback
    except Exception as exc:
        logger.warning(
            "OpenRouter query transformation failed; using original query model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )
        return query
