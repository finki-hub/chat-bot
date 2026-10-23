import logging

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.data.connection import Database
from app.llms.context import get_retrieved_context_with_sources
from app.llms.models import ChatModel, Model
from app.llms.query_modes import QueryTransformMode
from app.llms.retrieval_result import RetrievalSourcePayload

logger = logging.getLogger(__name__)

SEARCH_TOOL_NAME = "search_knowledge_base"
_SEARCH_TOOL_DESCRIPTION = (
    "Пребарај ја базата на знаења на ФИНКИ (FAQ и официјални документи: правилници, "
    "конкурси, ценовници, упатства). Враќа најрелевантни извори за барањето. "
    "Испрати кратко барање на македонски, со кирилица, со клучните поими."
)
_NO_RESULTS = "Нема пронајдени извори за ова барање."
_LIMIT_REACHED = (
    "Достигнат е лимитот на пребарувања за овој одговор. "
    "Одговори со веќе пронајдените извори."
)


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
    transform_model: ChatModel,
    *,
    max_searches: int,
) -> list[BaseTool]:
    searches = 0

    async def search_knowledge_base(
        query: str,
    ) -> tuple[str, dict[str, object]]:
        nonlocal searches
        # Parallel calls from one model turn all pass through here, so the cap counts
        # them too; the prompt's limit alone was exceeded in testing.
        if searches >= max_searches:
            return _LIMIT_REACHED, {"summary": "Достигнат лимит на пребарувања"}
        searches += 1
        retrieved = await get_retrieved_context_with_sources(
            db=db,
            query=query,
            embedding_model=embedding_model,
            query_transform_model=transform_model,
            query_transform_mode=QueryTransformMode.RAW,
        )
        sources: list[RetrievalSourcePayload] = retrieved.sources_payload()
        hit = bool(retrieved.text)
        logger.info(
            "agent.search query_len=%d hit=%s sources=%d context_chars=%d",
            len(query),
            hit,
            len(sources),
            len(retrieved.text),
        )
        return retrieved.text or _NO_RESULTS, {
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
