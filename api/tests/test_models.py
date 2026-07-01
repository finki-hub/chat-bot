from app.llms.models import (
    ANTHROPIC_QUERY_TRANSFORM_MODELS,
    CHAT_MODELS,
    REASONING_CAPABLE_MODELS,
    Model,
)


def test_claude_sonnet_5_is_a_supported_anthropic_chat_model():
    model = Model("claude-sonnet-5")

    assert model in CHAT_MODELS
    assert model in ANTHROPIC_QUERY_TRANSFORM_MODELS
    assert model in REASONING_CAPABLE_MODELS
