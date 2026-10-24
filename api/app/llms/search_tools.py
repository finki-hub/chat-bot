import asyncio
import logging
from dataclasses import dataclass

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from app.data.connection import Database
from app.llms.context import get_retrieved_context_with_sources
from app.llms.link_context import RelevantLinks
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
_SEARCH_FAILED = "Пребарувањето не успеа. Обиди се со поинакво барање."


class SearchKnowledgeBaseSchema(BaseModel):
    query: str = Field(
        description="Кратко барање на македонски (кирилица) со клучните поими.",
        min_length=1,
        max_length=500,
    )


@dataclass(frozen=True, slots=True)
class SearchResult:
    content: str
    sources: list[RetrievalSourcePayload]
    hit: bool


def _search_summary(source_count: int, *, hit: bool) -> str:
    if not hit:
        return "Нема пронајдени извори"
    if source_count == 0:
        return "Пронајдени се делумно поврзани извори"
    return f"Пронајдени извори: {source_count}"


class KnowledgeSearch:
    """Search FAQs, documents and relevant links for one answer."""

    def __init__(
        self,
        db: Database,
        embedding_model: Model,
        credentials: LlmProviderCredentials | None,
    ) -> None:
        self._db = db
        self._embedding_model = embedding_model
        self._credentials = credentials
        self._links = RelevantLinks(db)

    async def search(self, query: str) -> SearchResult:
        retrieved, links = await asyncio.gather(
            get_retrieved_context_with_sources(
                db=self._db,
                query=query,
                embedding_model=self._embedding_model,
                credentials=self._credentials,
            ),
            self._links.render(query),
        )
        sources = retrieved.sources_payload()
        content = "\n\n".join(part for part in (retrieved.text, links) if part)
        logger.info(
            "agent.search query_len=%d hit=%s sources=%d context_chars=%d links=%s",
            len(query),
            bool(retrieved.text),
            len(sources),
            len(retrieved.text),
            bool(links),
        )
        return SearchResult(content=content, sources=sources, hit=bool(retrieved.text))

    def tools(self) -> list[BaseTool]:
        async def search_knowledge_base(
            query: str,
        ) -> tuple[str, dict[str, object]]:
            try:
                result = await self.search(query)
            # The default tool error handler re-raises, which would end the whole answer;
            # a failed search becomes a failed tool result the model can work around.
            # ruff: ignore[BLE001] -- every search failure must reach the model as a result
            except Exception as exc:
                logger.warning(
                    "Knowledge search failed error_type=%s",
                    type(exc).__name__,
                )
                return _SEARCH_FAILED, {"failed": True}
            return result.content or _NO_RESULTS, {
                "hit": result.hit,
                "sources": result.sources,
                "summary": _search_summary(len(result.sources), hit=result.hit),
                "context_chars": len(result.content),
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
