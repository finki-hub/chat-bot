"""User-scoped Go clients; protocol routing is not an OpenAI credential alias.

Endpoints: https://opencode.ai/v2/docs/console/go
Reasoning/sampling capabilities: models.dev's opencode-go registry and current
vendor guides. Grok 4.7, GLM-5.3 and Kimi K3 cannot disable thinking; an off
request selects low effort rather than retaining their high/max defaults.
"""

import logging
from collections.abc import Generator
from typing import Any, override

import openai
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.llms.agents import (
    StreamObservation,
    capture_model_fallback,
    content_to_text,
    create_agent_token_generator,
    stream_sync_gen_as_sse,
)
from app.llms.models import (
    OPENCODE_GO_NO_SAMPLING_MODELS,
    OPENCODE_GO_QUERY_TRANSFORM_MODELS,
    Model,
)
from app.llms.prompts import build_agent_messages
from app.llms.provider_credentials import require_provider_credential
from app.llms.tools import agent_middleware, fallback_prompts, get_agent_tools
from app.schemas.chat_credentials import (
    OPENCODE_GO_DEFAULT_BASE_URL,
    ChatCredentialSecret,
)

logger = logging.getLogger(__name__)

_RESPONSES_MODELS = frozenset(
    {Model.OPENCODE_GO_GPT_6_LUNA, Model.OPENCODE_GO_GROK_4_7}
)
_MESSAGES_MODELS = frozenset(
    {
        Model.OPENCODE_GO_CLAUDE_HAIKU_5_5,
        Model.OPENCODE_GO_MINIMAX_M3,
        Model.OPENCODE_GO_QWEN3_8_MAX,
    }
)
_DEEPSEEK_MODELS = frozenset(
    {
        Model.OPENCODE_GO_DEEPSEEK_V4_1_FLASH,
        Model.OPENCODE_GO_DEEPSEEK_V4_PRO,
    }
)


class GoChatCompletions(ChatOpenAI):
    """Preserve compatible-provider reasoning across streaming and tool rounds.

    ChatOpenAI intentionally drops non-OpenAI reasoning_content. These small
    conversion hooks retain it without changing shared stream/cancellation code.
    """

    @override
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        result = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if result is not None and isinstance(result.message, AIMessageChunk):
            result.message.response_metadata["model_provider"] = "opencode-go"
            if choices and isinstance(
                reasoning := (choices[0].get("delta") or {}).get("reasoning_content"),
                str,
            ):
                result.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    @override
    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(
            result.generations, payload["choices"], strict=True
        ):
            generation.message.response_metadata["model_provider"] = "opencode-go"
            if isinstance(reasoning := choice["message"].get("reasoning_content"), str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    @override
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for message, wire in zip(messages, payload["messages"], strict=True):
            if isinstance(message, AIMessage) and isinstance(
                reasoning := message.additional_kwargs.get("reasoning_content"), str
            ):
                wire["reasoning_content"] = reasoning
        return payload


def get_opencode_go_llm(
    model: Model,
    temperature: float,
    top_p: float,
    max_tokens: int,
    *,
    reasoning: bool = False,
    credential: ChatCredentialSecret | None = None,
    conversation_id: str | None = None,
) -> ChatOpenAI | ChatAnthropic:
    credential = require_provider_credential("opencode-go", credential)
    if credential.provider != "opencode-go":
        raise ValueError("OpenCode Go requires its own provider credential")
    if not credential.api_key.strip():
        raise ValueError("OpenCode Go requires a nonblank user key")
    if model not in OPENCODE_GO_QUERY_TRANSFORM_MODELS:
        raise ValueError("Unsupported OpenCode Go model")
    if not conversation_id:
        raise ValueError("OpenCode Go requires a conversation ID")
    if model == Model.OPENCODE_GO_QWEN3_8_MAX and reasoning and max_tokens <= 1024:
        raise ValueError(
            "OpenCode Go Qwen reasoning requires max_tokens greater than 1024"
        )
    headers = {"User-Agent": "finki-hub/1.0", "x-opencode-session": conversation_id}
    base_url = (credential.base_url or OPENCODE_GO_DEFAULT_BASE_URL).rstrip("/")
    # Store one OpenAI-compatible base. Anthropic appends /v1/messages itself.
    root_url = base_url.removesuffix("/v1")
    options: dict[str, object] = {
        "model": model.value.removeprefix("opencode-go:"),
        "api_key": SecretStr(credential.api_key),
        "base_url": root_url if model in _MESSAGES_MODELS else f"{root_url}/v1",
        "default_headers": headers,
        "max_tokens": max_tokens,
        "streaming": True,
        "timeout": 120.0,
    }
    if model not in OPENCODE_GO_NO_SAMPLING_MODELS:
        options["temperature"] = temperature
    else:
        options["temperature"] = None
    if model in _MESSAGES_MODELS:
        # Messages compatibility does not imply identical model-specific knobs.
        # Never send OpenRouter's reasoning envelope.
        if model == Model.OPENCODE_GO_CLAUDE_HAIKU_5_5:
            options["thinking"] = (
                {"type": "adaptive", "display": "summarized"}
                if reasoning
                else {"type": "disabled"}
            )
            options["effort"] = "medium"
        elif model == Model.OPENCODE_GO_MINIMAX_M3:
            # https://platform.minimax.io/docs/api-reference/text-anthropic-api
            # M3 uses adaptive (not enabled/budget_tokens) and supports both
            # temperature and top_p, including when thinking is requested.
            options["thinking"] = {"type": "adaptive" if reasoning else "disabled"}
            options["top_p"] = top_p
        elif reasoning:
            options["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(4096, max_tokens - 1),
            }
            options["temperature"] = None
        else:
            options["thinking"] = {"type": "disabled"}
        return ChatAnthropic.model_validate(options)
    options["use_responses_api"] = model in _RESPONSES_MODELS
    options["stream_usage"] = True
    if model in _RESPONSES_MODELS:
        options["store"] = False
        if reasoning:
            options["reasoning"] = (
                {"effort": "medium", "summary": "auto"}
                if model == Model.OPENCODE_GO_GPT_6_LUNA
                else {"effort": "high"}
            )
        elif model == Model.OPENCODE_GO_GPT_6_LUNA:
            options["reasoning"] = {"effort": "none"}
        else:
            # https://docs.x.ai/developers/model-capabilities/text/reasoning
            # Grok 4.7 cannot disable reasoning.
            # An off request selects the lowest documented effort instead.
            options["reasoning"] = {"effort": "low"}
        return ChatOpenAI.model_validate(options)
    if model not in OPENCODE_GO_NO_SAMPLING_MODELS:
        options["top_p"] = top_p
    if reasoning:
        options["reasoning_effort"] = (
            "max" if model == Model.OPENCODE_GO_KIMI_K3 else "high"
        )
    elif model in _DEEPSEEK_MODELS:
        # https://api-docs.deepseek.com/api/create-chat-completion
        # DeepSeek defaults to enabled/high when effort is omitted. Go's
        # compatible route accepts the same explicit none disable contract.
        options["reasoning_effort"] = "none"
    else:
        # https://docs.z.ai/guides/capabilities/thinking-mode
        # https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
        # GLM-5.3 and Kimi K3 always think per their current vendor guides.
        # Omission would retain default high/max thinking, not disable it.
        options["reasoning_effort"] = "low"
    return GoChatCompletions.model_validate(options)


async def stream_opencode_go_agent_response(
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
    conversation_id: str | None = None,
) -> StreamingResponse:
    llm = get_opencode_go_llm(
        model,
        temperature,
        top_p,
        max_tokens,
        reasoning=reasoning,
        credential=credential,
        conversation_id=conversation_id,
    )
    messages = build_agent_messages(system_prompt, history or [], user_prompt)
    try:
        agent = create_agent(
            llm,
            await get_agent_tools(),
            middleware=agent_middleware(),
        )
        return StreamingResponse(
            create_agent_token_generator(agent, messages, observation),
            media_type="text/event-stream",
        )
    # ruff: ignore[BLE001] -- setup fallback stays on the same scoped client
    except Exception as exc:
        logger.warning(
            "OpenCode Go agent setup failed; using regular response model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )
        capture_model_fallback(
            observation,
            from_model=model.value,
            to_model=model.value,
            reason="agent_setup_failed",
        )

        fallback_system, fallback_user = await fallback_prompts(
            system_prompt,
            user_prompt,
        )
        fallback_messages = build_agent_messages(
            fallback_system,
            history or [],
            fallback_user,
        )

        def chunks() -> Generator[AIMessageChunk]:
            yield from llm.stream(fallback_messages)

        return stream_sync_gen_as_sse(chunks())


async def transform_query_with_opencode_go(
    query: str,
    model: Model,
    *,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    credential: ChatCredentialSecret | None = None,
    conversation_id: str | None = None,
) -> str:
    try:
        llm = get_opencode_go_llm(
            model,
            temperature,
            top_p,
            max_tokens,
            credential=credential,
            conversation_id=conversation_id,
        )
        response = await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=query)]
        )
        return content_to_text(response.content).strip()
    # ruff: ignore[BLE001] -- query/title generation is optional enrichment
    except Exception as exc:
        logger.warning(
            "OpenCode Go query transformation failed; using original query model=%s error_type=%s",
            model.value,
            type(exc).__name__,
        )
        return query
