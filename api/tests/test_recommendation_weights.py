from datetime import UTC, datetime, timedelta

import pytest

from app.schemas.recommendations import RecommendationRequestSchema
from app.services import recommendations
from app.services.recommendations import _get_mentor_prior, _scoring_weights


def test_include_prior_defaults_to_true() -> None:
    payload = RecommendationRequestSchema(title="Систем за препорака")

    assert payload.include_prior is True


def test_student_facing_request_disables_prior_signals() -> None:
    payload = RecommendationRequestSchema(
        title="Систем за препорака",
        include_prior=False,
    )

    weights = _scoring_weights(payload)

    assert weights.mentor_prior_weight == 0.0
    assert weights.pair_affinity_weight == 0.0
    assert weights.expertise_weight > 0.0


def test_default_request_keeps_prior_signals() -> None:
    payload = RecommendationRequestSchema(title="Систем за препорака")

    weights = _scoring_weights(payload)

    assert weights.mentor_prior_weight > 0.0
    assert weights.pair_affinity_weight > 0.0


@pytest.mark.anyio
async def test_member_prior_keeps_only_recent_committees(monkeypatch) -> None:
    today = datetime.now(UTC).date()

    async def committees(_db: object) -> list[dict[str, object]]:
        return [
            {
                "mentor": "Short",
                "member1": "Old",
                "member2": "Old Two",
                "date_of_submission": today - timedelta(days=4 * 365),
            },
            {
                "mentor": "Short",
                "member1": "New",
                "member2": "New Two",
                "date_of_submission": today,
            },
        ]

    monkeypatch.setattr(recommendations, "get_defended_committees", committees)
    monkeypatch.setattr(recommendations, "_mentor_prior_cache", {})

    mentor_prior, member_prior = await _get_mentor_prior(
        object(),  # type: ignore[arg-type]
        {"Short": "Full Name"},
    )

    assert set(mentor_prior.by_mentor["Full Name"]) == {
        "Old",
        "Old Two",
        "New",
        "New Two",
    }
    assert member_prior.by_mentor == {"Full Name": {"New": 1.0, "New Two": 1.0}}
