import logging

from fastapi.responses import StreamingResponse

from app.data.connection import Database
from app.llms.agents import StreamObservation
from app.llms.models import model_id
from app.llms.prompts import (
    AGENT_SEARCH_TOOLS_PROMPT,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    build_user_agent_prompt,
    markdown_instructions,
    to_history_messages,
)
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.recommendation_tools import build_recommendation_tools
from app.llms.search_tools import KnowledgeSearch
from app.llms.streams import stream_response_with_agent
from app.llms.tools import MAX_SEARCHES_PER_ANSWER, agent_request_tools
from app.schemas.chat import ChatSchema
from app.schemas.chat_credentials import ChatCredentialSecret
from app.utils.settings import Settings

logger = logging.getLogger(__name__)

settings = Settings()

PROMPT_SEPARATOR = "\n\n"
AGENT_SEARCH_CONTEXT = "Изворите не се пребарани однапред; пребарај ги со алатката."
NO_SOURCES = "Не можев да пронајдам релевантни информации во базата на податоци."


async def handle_chat(
    payload: ChatSchema,
    context: str,
    *,
    observation: StreamObservation | None = None,
    db: Database | None = None,
    inference_credential: ChatCredentialSecret | None = None,
    upstream_model: str | None = None,
    conversation_id: str | None = None,
    max_tokens: int | None = None,
    retrieval_credentials: LlmProviderCredentials | None = None,
) -> StreamingResponse:
    """
    Handle chat using an agent with MCP tool support.
    Falls back to regular streaming if no tools are available.
    """
    logger.info(
        "Handling chat for user prompt length: '%d' with model: %s",
        len(payload.query),
        model_id(payload.inference_model),
    )

    interface_format = markdown_instructions(payload.interface)
    system_prompt = PROMPT_SEPARATOR.join(
        [
            DEFAULT_AGENT_SYSTEM_PROMPT,
            AGENT_SEARCH_TOOLS_PROMPT.replace(
                "{max_searches}",
                str(MAX_SEARCHES_PER_ANSWER),
            ),
            interface_format,
        ],
    )
    user_prompt = build_user_agent_prompt(
        PROMPT_SEPARATOR.join([context, AGENT_SEARCH_CONTEXT]),
        payload.query,
    )
    history = to_history_messages(
        [
            (turn.role, turn.content)
            for turn in payload.capped_history(settings.CHAT_HISTORY_MAX_TURNS)
        ],
    )

    tools = build_recommendation_tools(db) if db is not None else []
    search = (
        KnowledgeSearch(db, payload.embeddings_model, retrieval_credentials)
        if db is not None
        else None
    )
    if search is not None:
        tools.extend(search.tools())

    async def fallback() -> tuple[str, str]:
        found = await search.search(payload.query) if search is not None else None
        sources = found.content if found is not None and found.content else NO_SOURCES
        return (
            PROMPT_SEPARATOR.join([DEFAULT_AGENT_SYSTEM_PROMPT, interface_format]),
            build_user_agent_prompt(
                PROMPT_SEPARATOR.join([context, sources]),
                payload.query,
            ),
        )

    with agent_request_tools(tools, fallback):
        return await stream_response_with_agent(
            user_prompt,
            payload.inference_model,
            system_prompt=system_prompt,
            history=history,
            temperature=payload.temperature,
            top_p=payload.top_p,
            max_tokens=payload.max_tokens if max_tokens is None else max_tokens,
            reasoning=payload.reasoning,
            observation=observation,
            interface=payload.interface,
            credential=inference_credential,
            upstream_model=upstream_model,
            conversation_id=conversation_id,
        )
