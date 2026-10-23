import asyncio
import logging

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.data.connection import Database
from app.llms.context import get_retrieved_context_with_sources
from app.llms.link_context import get_relevant_links_context
from app.llms.models import Model
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.retrieval_result import RetrievalSourcePayload
from app.llms.tools import SEARCH_TOOL_NAME

logger = logging.getLogger(__name__)

_SEARCH_TOOL_DESCRIPTION = (
    "Пребарај ја базата на знаења на ФИНКИ (FAQ и официјални документи: правилници, "
    "конкурси, ценовници, упатства) и корисните линкови. Враќа најрелевантни извори "
    "за барањето. Испрати кратко барање на македонски, со кирилица, со клучните поими."
)
_NO_RESULTS = "Нема пронајдени извори за ова барање."


class SearchKnowledgeBaseSchema(BaseModel):
    query: str = Field(
        description="Кратко барање на македонски (кирилица) со клучните поими.",
        min_length=1,
        max_length=500,
    )


def _search_summary(source_count: int, *, hit: bool) -> str:
    if not hit:
        return "Нема пронајдени извори"
    if source_count == 0:
        return "Пронајдени се делумно поврзани извори"
    return f"Пронајдени извори: {source_count}"


def build_search_tools(
    db: Database,
    embedding_model: Model,
    credentials: LlmProviderCredentials | None,
) -> list[BaseTool]:
    async def search_knowledge_base(
        query: str,
    ) -> tuple[str, dict[str, object]]:
        retrieved, links = await asyncio.gather(
            get_retrieved_context_with_sources(
                db=db,
                query=query,
                embedding_model=embedding_model,
                credentials=credentials,
            ),
            get_relevant_links_context(db, query),
        )
        sources: list[RetrievalSourcePayload] = retrieved.sources_payload()
        hit = bool(retrieved.text)
        logger.info(
            "agent.search query_len=%d hit=%s sources=%d context_chars=%d links=%s",
            len(query),
            hit,
            len(sources),
            len(retrieved.text),
            bool(links),
        )
        content = "\n\n".join(part for part in (retrieved.text, links) if part)
        return content or _NO_RESULTS, {
            "hit": hit,
            "sources": sources,
            "summary": _search_summary(len(sources), hit=hit),
        }

    return [
        StructuredTool.from_function(
            coroutine=search_knowledge_base,
            name=SEARCH_TOOL_NAME,
            description=_SEARCH_TOOL_DESCRIPTION,
            args_schema=SearchKnowledgeBaseSchema,
            response_format="content_and_artifact",
        ),
    ]
