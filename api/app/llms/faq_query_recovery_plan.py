import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from app.data.connection import Database
from app.data.questions import get_question_by_name_query, get_question_names_query
from app.llms.faq_query_recovery import (
    FaqQueryRecovery,
    is_faq_query_recovery_eligible,
    resolve_faq_query,
)
from app.schemas.questions import QuestionSchema

logger = logging.getLogger(__name__)

_FAQ_TITLE_CATALOG_CAP = 512
FaqQueryRecoveryOutcome = Literal["skipped", "no_match", "matched", "error"]


@dataclass(frozen=True, slots=True)
class FaqQueryRecoveryPlan:
    outcome: FaqQueryRecoveryOutcome
    recovery: FaqQueryRecovery | None = None
    question: QuestionSchema | None = None


async def plan_faq_query_recovery(
    db: Database,
    query: str,
) -> FaqQueryRecoveryPlan:
    """Resolve only against a complete, bounded FAQ title catalog and exact row."""
    if not is_faq_query_recovery_eligible(query):
        return FaqQueryRecoveryPlan(outcome="skipped")

    try:
        titles = await get_question_names_query(
            db,
            limit=_FAQ_TITLE_CATALOG_CAP + 1,
        )
        if len(titles) > _FAQ_TITLE_CATALOG_CAP:
            return FaqQueryRecoveryPlan(outcome="skipped")
        return await plan_faq_query_recovery_from_titles(db, query, titles)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # ruff: ignore[BLE001] -- local recovery is optional
        logger.warning(
            "Local FAQ query recovery unavailable error_type=%s",
            type(exc).__name__,
        )
        return FaqQueryRecoveryPlan(outcome="error")


async def plan_faq_query_recovery_from_titles(
    db: Database,
    query: str,
    titles: list[str] | tuple[str, ...],
) -> FaqQueryRecoveryPlan:
    """Resolve against a previously loaded, complete FAQ title catalog."""
    if len(titles) > _FAQ_TITLE_CATALOG_CAP:
        return FaqQueryRecoveryPlan(outcome="skipped")
    if not is_faq_query_recovery_eligible(query):
        return FaqQueryRecoveryPlan(outcome="skipped")
    try:
        recovery = resolve_faq_query(query, titles)
        if recovery is None:
            return FaqQueryRecoveryPlan(outcome="no_match")
        question = await get_question_by_name_query(db, recovery.title)
        if question is None:
            return FaqQueryRecoveryPlan(outcome="error")
        return FaqQueryRecoveryPlan(
            outcome="matched",
            recovery=recovery,
            question=question,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # ruff: ignore[BLE001] -- local recovery is optional
        logger.warning(
            "Local FAQ query recovery unavailable error_type=%s",
            type(exc).__name__,
        )
        return FaqQueryRecoveryPlan(outcome="error")
