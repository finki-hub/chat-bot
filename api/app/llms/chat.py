import logging

from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms.agents import StreamObservation
from app.llms.prompts import (
    DEFAULT_AGENT_SYSTEM_PROMPT,
    build_user_agent_prompt,
    markdown_instructions,
    to_history_messages,
)
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.recommendation_tools import build_recommendation_tools
from app.llms.streams import stream_response_with_agent
from app.llms.tools import agent_request_tools
from app.schemas.chat import ChatSchema
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()


async def handle_chat(
    payload: ChatSchema,
    context: str,
    observation: StreamObservation | None = None,
    db: Database | None = None,
    credentials: LlmProviderCredentials | None = None,
) -> StreamingResponse:
    """
    Handle chat using an agent with MCP tool support.
    Falls back to regular streaming if no tools are available.
    """
    logger.info(
        "Handling chat for user prompt length: '%d' with model: %s",
        len(payload.query),
        payload.inference_model.value,
    )

    system_prompt = "\n\n".join(
        [
            DEFAULT_AGENT_SYSTEM_PROMPT,
            markdown_instructions(payload.interface),
        ],
    )
    user_prompt = build_user_agent_prompt(context, payload.query)
    history = to_history_messages(
        [
            (turn.role, turn.content)
            for turn in payload.capped_history(settings.CHAT_HISTORY_MAX_TURNS)
        ],
    )

    tools = build_recommendation_tools(db) if db is not None else []
    with agent_request_tools(tools):
        return await stream_response_with_agent(
            user_prompt,
            payload.inference_model,
            system_prompt=system_prompt,
            history=history,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens,
            reasoning=payload.reasoning,
            observation=observation,
            interface=payload.interface,
            credentials=credentials,
        )
