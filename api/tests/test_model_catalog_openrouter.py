from app.llms.model_catalog_remote import parse_models_dev
from app.llms.models import Model


def test_models_dev_openrouter_metadata_matches_unprefixed_remote_id() -> None:
    payload = b"""{
        "openrouter": {
            "id": "openrouter",
            "name": "OpenRouter",
            "models": {
                "z-ai/glm-5.3": {
                    "id": "z-ai/glm-5.3",
                    "name": "Remote GLM 5.3",
                    "reasoning": true
                }
            }
        }
    }"""

    metadata = parse_models_dev(payload)

    assert metadata[Model.OPENROUTER_GLM_5_3].name == "Remote GLM 5.3"
