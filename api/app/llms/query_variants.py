import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, assert_never

from app.llms.models import ChatModel
from app.llms.prompts import HYDE_SYSTEM_PROMPT
from app.llms.provider_credentials import LlmProviderCredentials
from app.llms.query_modes import QueryTransformMode
from app.llms.query_normalization import romanized_macedonian_variant
from app.llms.query_transform import QUERY_TRANSFORM_MAX_TOKENS, transform_query
from app.utils.timing import QueryTransformFallbackReason, timed

QueryVariantKind = Literal["raw", "rewrite", "hyde", "transliteration"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QueryVariant:
    kind: QueryVariantKind
    text: str
    is_document: bool


@dataclass(frozen=True, slots=True)
class QueryVariantBundle:
    variants: tuple[QueryVariant, ...]
    rerank_query: str
    unchanged_kinds: frozenset[QueryVariantKind] = frozenset()

    def fallback_reason(
        self, requested: QueryTransformMode
    ) -> QueryTransformFallbackReason:
        """Classify retained variants; output identical to the query is not a failure."""
        if requested == QueryTransformMode.RAW:
            return "not_requested"
        if self.mode == requested:
            return "none"
        missing = _requested_kinds(requested) - _requested_kinds(self.mode)
        if missing <= self.unchanged_kinds:
            return "unchanged_variants"
        if self.mode == QueryTransformMode.RAW:
            return "no_usable_variants"
        return "partial_variants"

    @property
    def transliteration_variant_count(self) -> int:
        return sum(variant.kind == "transliteration" for variant in self.variants)

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
        raise AssertionError("Unreachable query variant mode")


def _requested_kinds(mode: QueryTransformMode) -> frozenset[QueryVariantKind]:
    match mode:
        case QueryTransformMode.RAW:
            return frozenset()
        case QueryTransformMode.REWRITE:
            return frozenset({"rewrite"})
        case QueryTransformMode.HYDE:
            return frozenset({"hyde"})
        case QueryTransformMode.REWRITE_HYDE:
            return frozenset({"rewrite", "hyde"})
        case unreachable:
            assert_never(unreachable)
    raise AssertionError(f"Unhandled query transform mode: {mode}")


def _usable(search_query: str, output: str | None) -> str | None:
    return output if output and output != search_query.strip() else None


def _unchanged_kinds(
    search_query: str,
    outputs: dict[QueryVariantKind, str | None],
) -> frozenset[QueryVariantKind]:
    return frozenset(
        kind for kind, output in outputs.items() if output == search_query.strip()
    )


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
    bundle = await _build_transform_variants(
        search_query,
        query_transform_model,
        mode,
        credentials,
    )
    transliterated = romanized_macedonian_variant(search_query)
    if transliterated is None:
        return bundle
    # Romanized queries embed far from the Cyrillic corpus, so dense retrieval must
    # not depend on an LLM transform being available to bridge the script gap.
    *transforms, raw = bundle.variants
    has_rewrite = any(variant.kind == "rewrite" for variant in transforms)
    return QueryVariantBundle(
        variants=(
            *transforms,
            QueryVariant(
                kind="transliteration", text=transliterated, is_document=False
            ),
            raw,
        ),
        rerank_query=bundle.rerank_query if has_rewrite else transliterated,
        unchanged_kinds=bundle.unchanged_kinds,
    )


async def _build_transform_variants(
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
            rewrite_output = await _rewrite_query(
                search_query,
                query_transform_model,
                credentials,
            )
            rewritten = _usable(search_query, rewrite_output)
            match rewritten:
                case None:
                    return QueryVariantBundle(
                        variants=(raw,),
                        rerank_query=search_query,
                        unchanged_kinds=_unchanged_kinds(
                            search_query, {"rewrite": rewrite_output}
                        ),
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
            hyde_output = await _hyde_passage(
                search_query,
                query_transform_model,
                credentials,
            )
            hyde = _usable(search_query, hyde_output)
            match hyde:
                case None:
                    return QueryVariantBundle(
                        variants=(raw,),
                        rerank_query=search_query,
                        unchanged_kinds=_unchanged_kinds(
                            search_query, {"hyde": hyde_output}
                        ),
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
            rewrite_output, hyde_output = await asyncio.gather(
                _rewrite_query(search_query, query_transform_model, credentials),
                _hyde_passage(search_query, query_transform_model, credentials),
            )
            rewritten = _usable(search_query, rewrite_output)
            hyde = _usable(search_query, hyde_output)
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
                unchanged_kinds=_unchanged_kinds(
                    search_query,
                    {"rewrite": rewrite_output, "hyde": hyde_output},
                ),
            )
        case unreachable:
            assert_never(unreachable)
    raise AssertionError(f"Unhandled query transform mode: {mode}")


async def _rewrite_query(
    search_query: str,
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None = None,
) -> str | None:
    try:
        with timed("retrieval.query_rewrite"):
            rewritten = await transform_query(
                search_query,
                query_transform_model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=QUERY_TRANSFORM_MAX_TOKENS,
                credentials=credentials,
            )
    # ruff: ignore[BLE001] -- rewrite generation is optional retrieval enrichment
    except Exception as exc:
        logger.warning(
            "Query rewrite failed; omitting variant error_type=%s",
            type(exc).__name__,
        )
        return None
    return rewritten.strip()


async def _hyde_passage(
    search_query: str,
    query_transform_model: ChatModel,
    credentials: LlmProviderCredentials | None = None,
) -> str | None:
    try:
        with timed("retrieval.hyde"):
            hyde = await transform_query(
                search_query,
                query_transform_model,
                system_prompt=HYDE_SYSTEM_PROMPT,
                temperature=0.2,
                top_p=1.0,
                max_tokens=QUERY_TRANSFORM_MAX_TOKENS,
                credentials=credentials,
            )
    # ruff: ignore[BLE001] -- HyDE generation is optional retrieval enrichment
    except Exception as exc:
        logger.warning(
            "HyDE generation failed; omitting variant error_type=%s",
            type(exc).__name__,
        )
        return None
    return hyde.strip()
