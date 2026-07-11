import json
from datetime import UTC, datetime
from uuid import uuid4


class FakeChatDatabase:
    def __init__(self) -> None:
        self.conversations: dict[object, dict[str, object]] = {}
        self.feedback: dict[tuple[object, object, object], dict[str, object]] = {}
        self.messages: dict[object, dict[str, object]] = {}
        self.users: dict[tuple[object, object], dict[str, object]] = {}
        self.credentials: dict[tuple[object, object], dict[str, object]] = {}
        self.now = datetime(2026, 7, 7, tzinfo=UTC)

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        if "INSERT INTO chat_user (" in query:
            provider, provider_subject, email, name, avatar_url = args
            user_key = (provider, provider_subject)
            current = self.users.get(user_key)
            if current is None:
                current = {
                    "id": uuid4(),
                    "provider": provider,
                    "provider_subject": provider_subject,
                    "email": email,
                    "name": name,
                    "avatar_url": avatar_url,
                    "created_at": self.now,
                    "updated_at": self.now,
                }
                self.users[user_key] = current
                return current
            current["email"] = email
            current["name"] = name
            current["avatar_url"] = avatar_url
            current["updated_at"] = self.now
            return current

        if "INSERT INTO chat_user_credential" in query:
            user_id, provider, encrypted_api_key, base_url = args
            credential_key = (user_id, provider)
            current = self.credentials.get(credential_key)
            if current is None:
                current = {
                    "user_id": user_id,
                    "provider": provider,
                    "encrypted_api_key": encrypted_api_key,
                    "base_url": base_url,
                    "created_at": self.now,
                    "updated_at": self.now,
                }
                self.credentials[credential_key] = current
                return current
            current["encrypted_api_key"] = encrypted_api_key
            current["base_url"] = base_url
            current["updated_at"] = self.now
            return current

        if "FROM chat_user_credential" in query and "encrypted_api_key" in query:
            user_id, provider = args
            return self.credentials.get((user_id, provider))

        if "SELECT" in query and "assistant.content AS answer_text" in query:
            response_id, user_id = args
            for assistant in self.messages.values():
                conversation = self.conversations.get(assistant["conversation_id"])
                if (
                    assistant["response_id"] != response_id
                    or assistant["role"] != "assistant"
                    or conversation is None
                    or str(conversation["user_id"]) != user_id
                ):
                    continue
                prior_questions = [
                    message
                    for message in self.messages.values()
                    if message["conversation_id"] == assistant["conversation_id"]
                    and message["role"] == "user"
                    and self._created_at(message) <= self._created_at(assistant)
                ]
                question = max(prior_questions, key=self._created_at, default=None)
                metadata = assistant["metadata"]
                inference_model = None
                if isinstance(metadata, dict):
                    inference_model = metadata.get("inferenceModel")
                return {
                    "answer_text": assistant["content"],
                    "inference_model": inference_model,
                    "question_text": None if question is None else question["content"],
                }
            return None

        if "INSERT INTO feedback" in query:
            (
                response_id,
                client,
                user_id,
                feedback_type,
                client_ref,
                channel_id,
                guild_id,
                question_text,
                answer_text,
                inference_model,
                embeddings_model,
                query_transform_model,
            ) = args
            feedback_key = (response_id, client, user_id)
            current = self.feedback.get(feedback_key)
            if current is None:
                current = {
                    "id": uuid4(),
                    "response_id": response_id,
                    "client": client,
                    "user_id": user_id,
                    "feedback_type": feedback_type,
                    "client_ref": client_ref,
                    "channel_id": channel_id,
                    "guild_id": guild_id,
                    "question_text": question_text,
                    "answer_text": answer_text,
                    "inference_model": inference_model,
                    "embeddings_model": embeddings_model,
                    "query_transform_model": query_transform_model,
                    "created_at": self.now,
                    "updated_at": self.now,
                }
                self.feedback[feedback_key] = current
            else:
                current["feedback_type"] = feedback_type
                current["updated_at"] = self.now
            return current

        if "UPDATE chat_message AS assistant" in query:
            response_id, user_id, feedback_type = args
            for assistant in self.messages.values():
                conversation = self.conversations.get(assistant["conversation_id"])
                if (
                    assistant["response_id"] != response_id
                    or assistant["role"] != "assistant"
                    or conversation is None
                    or str(conversation["user_id"]) != user_id
                ):
                    continue
                metadata = assistant["metadata"]
                parsed = json.loads(metadata) if isinstance(metadata, str) else metadata
                if not isinstance(parsed, dict):
                    parsed = {}
                parsed["feedback"] = feedback_type
                assistant["metadata"] = parsed
                assistant["updated_at"] = self.now
                return {"id": assistant["id"]}
            return None

        if "INSERT INTO chat_conversation" in query:
            conversation_id, user_id, model, title = args
            inserted = self._conversation_row(
                conversation_id=conversation_id,
                user_id=user_id,
                active_stream_id=None,
                active_response_id=None,
                active_status=None,
                model=model,
                title=title,
                updated_at=self.now,
            )
            self.conversations[conversation_id] = inserted
            return inserted

        if (
            "UPDATE chat_conversation" in query
            and "SET active_stream_id = NULL" in query
        ):
            conversation_id, user_id, active_stream_id = args
            current = self._owned_conversation(conversation_id, user_id)
            if current is None or current["active_stream_id"] != active_stream_id:
                return None
            current["active_stream_id"] = None
            current["active_response_id"] = None
            current["active_status"] = None
            current["updated_at"] = self.now
            return current

        if "UPDATE chat_conversation" in query and "SET active_stream_id = $3" in query:
            conversation_id, user_id, active_stream_id, active_response_id, status = (
                args
            )
            current = self._owned_conversation(conversation_id, user_id)
            if current is None:
                return None
            current["active_stream_id"] = active_stream_id
            current["active_response_id"] = active_response_id
            current["active_status"] = status
            current["updated_at"] = self.now
            return current

        if "UPDATE chat_conversation" in query and "active_status = 'stopped'" in query:
            conversation_id, user_id, active_stream_id = args
            current = self._owned_conversation(conversation_id, user_id)
            if current is None or current["active_stream_id"] != active_stream_id:
                return None
            current["active_status"] = "stopped"
            current["updated_at"] = self.now
            return current

        if "UPDATE chat_conversation" in query and "model = COALESCE" in query:
            conversation_id, user_id, model, title, status = args
            current = self._owned_conversation(conversation_id, user_id)
            if current is None:
                return None
            if model is not None:
                current["model"] = model
            if title is not None:
                current["title"] = title
            if status is not None:
                current["active_status"] = status
            current["updated_at"] = self.now
            return current

        if "SELECT user_id FROM chat_conversation" in query:
            (conversation_id,) = args
            current = self.conversations.get(conversation_id)
            return None if current is None else {"user_id": current["user_id"]}

        if "SELECT * FROM chat_conversation" in query:
            conversation_id, user_id = args
            return self._owned_conversation(conversation_id, user_id)

        if "DELETE FROM chat_conversation" in query:
            conversation_id, user_id = args
            current = self._owned_conversation(conversation_id, user_id)
            if current is None:
                return None
            deleted = self.conversations.pop(conversation_id)
            stale_message_ids = [
                message_id
                for message_id, message in self.messages.items()
                if message["conversation_id"] == conversation_id
            ]
            for message_id in stale_message_ids:
                del self.messages[message_id]
            return deleted

        if "ON CONFLICT (conversation_id, response_id)" in query:
            conversation_id, response_id, message_id, content, metadata_json = args
            for existing in self.messages.values():
                if (
                    existing["conversation_id"] == conversation_id
                    and existing["response_id"] == response_id
                    and existing["role"] == "assistant"
                ):
                    existing["content"] = content
                    existing["metadata"] = metadata_json
                    existing["updated_at"] = self.now
                    return existing
            inserted_assistant = {
                "id": message_id,
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": content,
                "response_id": response_id,
                "metadata": metadata_json,
                "created_at": self.now,
                "updated_at": self.now,
            }
            self.messages[message_id] = inserted_assistant
            return inserted_assistant

        if "WITH target AS" in query and "DELETE FROM chat_message stale" in query:
            (
                message_id,
                conversation_id,
                user_id,
                content,
                response_id,
                metadata_json,
                retained_message_ids,
            ) = args
            target = self.messages.get(message_id)
            conversation = self.conversations.get(conversation_id)
            if (
                target is None
                or conversation is None
                or conversation["user_id"] != user_id
                or target["conversation_id"] != conversation_id
                or target["role"] != "assistant"
                or not isinstance(retained_message_ids, list)
            ):
                return None
            retained_ids = set(retained_message_ids)
            stale_message_ids = [
                stale_id
                for stale_id, stale in self.messages.items()
                if stale["conversation_id"] == conversation_id
                and self._created_at(stale) > self._created_at(target)
                and stale_id not in retained_ids
            ]
            target["content"] = content
            target["response_id"] = response_id
            target["metadata"] = metadata_json
            target["updated_at"] = self.now
            for stale_id in stale_message_ids:
                del self.messages[stale_id]
            return target

        if "INSERT INTO chat_message" in query:
            message_id, conversation_id, role, content, response_id, metadata_json = (
                args
            )
            current_message = self.messages.get(message_id)
            if current_message is None:
                inserted_message = {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "role": role,
                    "content": content,
                    "response_id": response_id,
                    "metadata": metadata_json,
                    "created_at": self.now,
                    "updated_at": self.now,
                }
                self.messages[message_id] = inserted_message
                return inserted_message
            if current_message["conversation_id"] != conversation_id:
                return None
            current_message["role"] = role
            current_message["content"] = content
            current_message["response_id"] = response_id
            current_message["metadata"] = metadata_json
            current_message["updated_at"] = self.now
            return current_message

        msg = f"Unhandled fetchrow query: {query}"
        raise AssertionError(msg)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        if "FROM chat_user_credential" in query:
            (user_id,) = args
            rows = [
                row for row in self.credentials.values() if row["user_id"] == user_id
            ]
            return sorted(rows, key=lambda row: str(row["provider"]))

        if "DELETE FROM chat_conversation" in query:
            (user_id,) = args
            deleted = [
                row for row in self.conversations.values() if row["user_id"] == user_id
            ]
            deleted_ids = {row["id"] for row in deleted}
            for conversation_id in deleted_ids:
                del self.conversations[conversation_id]
            stale_message_ids = [
                message_id
                for message_id, message in self.messages.items()
                if message["conversation_id"] in deleted_ids
            ]
            for message_id in stale_message_ids:
                del self.messages[message_id]
            return deleted

        if "WHERE user_id = $1" in query:
            user_id = args[0]
            limit = args[1]
            if not isinstance(limit, int):
                raise AssertionError("conversation list limit must be int")
            rows = [
                row for row in self.conversations.values() if row["user_id"] == user_id
            ]
            return sorted(rows, key=self._updated_at, reverse=True)[:limit]

        if "FROM chat_message" in query:
            (conversation_id,) = args
            rows = [
                row
                for row in self.messages.values()
                if row["conversation_id"] == conversation_id
            ]
            return sorted(rows, key=self._created_at)

        msg = f"Unhandled fetch query: {query}"
        raise AssertionError(msg)

    async def fetchval(self, query: str, *args: object, column: int = 0) -> int:
        if "WITH cleared AS" not in query:
            msg = f"Unhandled fetchval query: {query}"
            raise AssertionError(msg)
        (stale_before,) = args
        if not isinstance(stale_before, datetime):
            raise TypeError("stale cutoff must be datetime")
        cleared = 0
        for row in self.conversations.values():
            if (
                row["active_stream_id"] is not None
                and self._updated_at(row) < stale_before
            ):
                row["active_stream_id"] = None
                row["active_response_id"] = None
                row["active_status"] = None
                cleared += 1
        return cleared

    async def execute(self, query: str, *args: object) -> str:
        if "DELETE FROM chat_user_credential" in query:
            user_id, provider = args
            self.credentials.pop((user_id, provider), None)
            return "DELETE 1"

        msg = f"Unhandled execute query: {query}"
        raise AssertionError(msg)

    def _created_at(self, row: dict[str, object]) -> datetime:
        created_at = row["created_at"]
        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be datetime")
        return created_at

    def _updated_at(self, row: dict[str, object]) -> datetime:
        updated_at = row["updated_at"]
        if not isinstance(updated_at, datetime):
            raise TypeError("updated_at must be datetime")
        return updated_at

    def _conversation_row(
        self,
        *,
        conversation_id: object,
        user_id: object,
        active_stream_id: object,
        active_response_id: object,
        active_status: object,
        model: object,
        title: object,
        updated_at: datetime,
    ) -> dict[str, object]:
        return {
            "id": conversation_id,
            "user_id": user_id,
            "active_stream_id": active_stream_id,
            "active_response_id": active_response_id,
            "active_status": active_status,
            "model": model,
            "title": title,
            "created_at": self.now,
            "updated_at": updated_at,
        }

    def _owned_conversation(
        self,
        conversation_id: object,
        user_id: object,
    ) -> dict[str, object] | None:
        current = self.conversations.get(conversation_id)
        if current is None or current["user_id"] != user_id:
            return None
        return current
