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

    @property
    def mode(self) -> QueryTransformMode:
        kinds = {variant.kind for variant in self.variants}
        match ("rewrite" in kinds, "hyde" in kinds):
            case False, False:
                return QueryTransformMode.RAW
            case True, False:
                return QueryTransformMode.REWRITE
            case False, True:
                return QueryTransformMode.HYDE
            case True, True:
                return QueryTransformMode.REWRITE_HYDE


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
            match rewritten:
                case None:
                    return QueryVariantBundle(
                        variants=(raw,),
                        rerank_query=search_query,
                    )
                case str():
                    return QueryVariantBundle(
                        variants=(
                            QueryVariant(
                                kind="rewrite",
                                text=rewritten,
                                is_document=False,
                            ),
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
            match hyde:
                case None:
                    return QueryVariantBundle(
                        variants=(raw,),
                        rerank_query=search_query,
                    )
                case str():
                    return QueryVariantBundle(
                        variants=(
                            QueryVariant(kind="hyde", text=hyde, is_document=True),
                            raw,
                        ),
                        rerank_query=search_query,
                    )
        case QueryTransformMode.REWRITE_HYDE:
            rewritten, hyde = await asyncio.gather(
                _rewrite_query(search_query, query_transform_model, credentials),
                _hyde_passage(search_query, query_transform_model, credentials),
            )
            variants = (
                *(
                    (QueryVariant(kind="hyde", text=hyde, is_document=True),)
                    if hyde is not None
                    else ()
                ),
                *(
                    (
                        QueryVariant(
                            kind="rewrite",
                            text=rewritten,
                            is_document=False,
                        ),
                    )
                    if rewritten is not None
                    else ()
                ),
                raw,
            )
            return QueryVariantBundle(
                variants=variants,
                rerank_query=rewritten or search_query,
            )
        case unreachable:
            assert_never(unreachable)
    raise AssertionError(f"Unhandled query transform mode: {mode}")


async def _rewrite_query(
    search_query: str,
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None = None,
) -> str | None:
    with timed("retrieval.query_rewrite"):
        rewritten = await transform_query(
            search_query,
            query_transform_model,
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            credentials=credentials,
        )
    rewritten = rewritten.strip()
    return rewritten if rewritten and rewritten != search_query.strip() else None


async def _hyde_passage(
    search_query: str,
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None = None,
) -> str | None:
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
    hyde = hyde.strip()
    return hyde if hyde and hyde != search_query.strip() else None
