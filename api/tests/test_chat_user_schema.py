from app.schemas.chat_user import ChatUser, ChatUserUpsert


def test_chat_user_upsert_normalizes_provider_before_literal_validation() -> None:
    # Given: a supported provider with surrounding transport whitespace.
    payload = {
        "provider": " discord ",
        "provider_subject": " discord-user-1 ",
    }

    # When: the request schema parses the payload.
    user = ChatUserUpsert.model_validate(payload)

    # Then: provider validation sees the normalized value.
    assert user.provider == "discord"
    assert user.provider_subject == "discord-user-1"


def test_chat_user_response_exposes_supported_providers_in_json_schema() -> None:
    # Given: the response model used by the chat user endpoint.
    schema = ChatUser.model_json_schema()

    # When: its provider property is rendered for OpenAPI.
    provider_schema = schema["properties"]["provider"]

    # Then: clients receive the same closed provider set as the request model.
    assert provider_schema["enum"] == ["google", "microsoft-entra-id", "discord"]
