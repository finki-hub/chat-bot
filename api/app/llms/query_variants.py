import asyncio
from dataclasses import dataclass
from typing import Literal, assert_never

from app.llms.models import ChatModel
from app.llms.prompts import HYDE_SYSTEM_PROMPT
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.query_modes import QueryTransformMode
from app.llms.query_transform import transform_query
from app.utils.timing import timed

QueryVariantKind = Literal["raw", "rewrite", "hyde"]


@dataclass(frozen=True, slots=True)
class QueryVariant:
    kind: QueryVariantKind
    text: str
    is_document: bool


@dataclass(frozen=True, slots=True)
class QueryVariantBundle:
    variants: tuple[QueryVariant, ...]
    rerank_query: str


def query_variant_count(mode: QueryTransformMode) -> int:
    match mode:
        case QueryTransformMode.RAW:
            return 1
        case QueryTransformMode.REWRITE | QueryTransformMode.HYDE:
            return 2
        case QueryTransformMode.REWRITE_HYDE:
            return 3
        case unreachable:
            assert_never(unreachable)
    raise AssertionError(f"Unhandled query transform mode: {mode}")


async def build_query_variants(
    search_query: str,
    query_transform_model: ChatModel,
    mode: QueryTransformMode,
    credentials: LlmProviderCredentials | None = None,
) -> QueryVariantBundle:
    raw = QueryVariant(kind="raw", text=search_query, is_document=False)

    match mode:
        case QueryTransformMode.RAW:
            return QueryVariantBundle(variants=(raw,), rerank_query=search_query)
        case QueryTransformMode.REWRITE:
            rewritten = await _rewrite_query(
                search_query,
                query_transform_model,
                credentials,
            )
            return QueryVariantBundle(
                variants=(
                    QueryVariant(kind="rewrite", text=rewritten, is_document=False),
                    raw,
                ),
                rerank_query=rewritten,
            )
        case QueryTransformMode.HYDE:
            hyde = await _hyde_passage(
                search_query,
                query_transform_model,
                credentials,
            )
            return QueryVariantBundle(
                variants=(QueryVariant(kind="hyde", text=hyde, is_document=True), raw),
                rerank_query=search_query,
            )
        case QueryTransformMode.REWRITE_HYDE:
            rewritten, hyde = await asyncio.gather(
                _rewrite_query(search_query, query_transform_model, credentials),
                _hyde_passage(search_query, query_transform_model, credentials),
            )
            return QueryVariantBundle(
                variants=(
                    QueryVariant(kind="hyde", text=hyde, is_document=True),
                    QueryVariant(kind="rewrite", text=rewritten, is_document=False),
                    raw,
                ),
                rerank_query=rewritten,
            )
        case unreachable:
            assert_never(unreachable)
    raise AssertionError(f"Unhandled query transform mode: {mode}")


async def _rewrite_query(
    search_query: str,
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None = None,
) -> str:
    with timed("retrieval.query_rewrite"):
        rewritten = await transform_query(
            search_query,
            query_transform_model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credentials=credentials,
        )
    return rewritten.strip() or search_query


async def _hyde_passage(
    search_query: str,
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None = None,
) -> str:
    with timed("retrieval.hyde"):
        hyde = await transform_query(
            search_query,
            query_transform_model,
            system_prompt=HYDE_SYSTEM_PROMPT,
            temperature=0.2,
            top_p=1.0,
            max_tokens=200,
            credentials=credentials,
        )
    return hyde.strip() or search_query
