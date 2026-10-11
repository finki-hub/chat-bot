import pytest

from app.schemas.chat import ChatSchema
from app.schemas.chat_title import ChatTitleSchema

RETIRED_MODEL_IDS = (
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-haiku-4-5",
    "gemini-3.1-flash-lite",
    "openrouter:x-ai/grok-4.6",
    "gpt-5.5",
    "gpt-5.4",
    "gemini-3.5-flash",
    "openrouter:deepseek/deepseek-v4-flash",
    "openrouter:deepseek/deepseek-v4-flash-0731",
    "openrouter:qwen/qwen3.8-max",
)


def test_chat_schema_accepts_dynamic_ollama_chat_model_tag():
    # Given: an installed Ollama model tag returned by the user's Ollama instance.
    ollama_model = "bge-m3:latest"

    # When: the model is used for generation and query transformation.
    payload = ChatSchema.model_validate(
        {
            "inference_model": ollama_model,
            "messages": [{"role": "user", "content": "hi"}],
            "query_transform_model": ollama_model,
        },
    )

    # Then: the dynamic tag is preserved instead of being forced through the enum.
    assert payload.inference_model == ollama_model
    assert payload.query_transform_model == ollama_model


def test_chat_schema_rejects_unknown_non_ollama_chat_model_id():
    with pytest.raises(
        ValueError,
        match="inference_model must be an active chat model",
    ):
        ChatSchema.model_validate(
            {
                "inference_model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )


@pytest.mark.parametrize("model_id", RETIRED_MODEL_IDS)
def test_chat_schema_rejects_retired_inference_model_ids(model_id: str) -> None:
    with pytest.raises(
        ValueError, match="inference_model must be an active chat model"
    ):
        ChatSchema.model_validate(
            {
                "inference_model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )


@pytest.mark.parametrize("model_id", RETIRED_MODEL_IDS)
def test_chat_schema_rejects_retired_query_transform_model_ids(model_id: str) -> None:
    with pytest.raises(
        ValueError,
        match="query_transform_model must be a chat-capable model",
    ):
        ChatSchema.model_validate(
            {
                "query_transform_model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )


@pytest.mark.parametrize("model_id", RETIRED_MODEL_IDS)
@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("provider_model", "provider_model must be a chat-capable model"),
        ("query_transform_model", "query_transform_model must be a chat-capable model"),
    ],
)
def test_chat_title_schema_rejects_retired_model_ids(
    model_id: str,
    field: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        ChatTitleSchema.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                field: model_id,
            },
        )


@pytest.mark.parametrize(
    "model_id",
    [":latest", "llama3.2:", "llama 3:latest", "llama3:latest:extra"],
)
def test_chat_schema_rejects_invalid_ollama_model_tags(model_id: str):
    with pytest.raises(
        ValueError,
        match="inference_model must be an active chat model",
    ):
        ChatSchema.model_validate(
            {
                "inference_model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )


def test_chat_schema_rejects_more_than_10_messages():
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(11)
    ]
    messages[-1] = {"role": "user", "content": "last"}

    with pytest.raises(ValueError, match="List should have at most 10 items"):
        ChatSchema.model_validate({"messages": messages})


def test_chat_schema_rejects_turns_over_2000_chars():
    with pytest.raises(ValueError, match="String should have at most 2000 characters"):
        ChatSchema.model_validate(
            {"messages": [{"role": "user", "content": "x" * 2001}]},
        )
