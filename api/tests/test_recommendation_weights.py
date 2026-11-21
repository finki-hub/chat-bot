from app.schemas.recommendations import RecommendationRequestSchema
from app.services.recommendations import _scoring_weights


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
