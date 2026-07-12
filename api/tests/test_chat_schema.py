import pytest

from app.schemas.chat import ChatSchema


def test_chat_schema_accepts_dynamic_ollama_chat_model_tag():
    # Given: an installed Ollama model tag returned by the user's Ollama instance.
    ollama_model = "bge-m3:latest"

    # When: the model is used for generation and query transformation.
    payload = ChatSchema(
        inference_model=ollama_model,
        messages=[{"role": "user", "content": "hi"}],
        query_transform_model=ollama_model,
    )

    # Then: the dynamic tag is preserved instead of being forced through the enum.
    assert payload.inference_model == ollama_model
    assert payload.query_transform_model == ollama_model


def test_chat_schema_rejects_unknown_non_ollama_chat_model_id():
    with pytest.raises(
        ValueError,
        match="inference_model must be an active chat model",
    ):
        ChatSchema(
            inference_model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
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
        ChatSchema(
            inference_model=model_id,
            messages=[{"role": "user", "content": "hi"}],
        )


def test_chat_schema_rejects_more_than_10_messages():
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(11)
    ]
    messages[-1] = {"role": "user", "content": "last"}

    with pytest.raises(ValueError, match="List should have at most 10 items"):
        ChatSchema(messages=messages)


def test_chat_schema_rejects_turns_over_2000_chars():
    with pytest.raises(ValueError, match="String should have at most 2000 characters"):
        ChatSchema(messages=[{"role": "user", "content": "x" * 2001}])
