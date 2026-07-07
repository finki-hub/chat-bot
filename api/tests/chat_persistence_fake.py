from datetime import UTC, datetime
from uuid import uuid4


class FakeChatDatabase:
    def __init__(self) -> None:
        self.conversations: dict[object, dict[str, object]] = {}
        self.messages: dict[object, dict[str, object]] = {}
        self.users: dict[tuple[object, object], dict[str, object]] = {}
        self.now = datetime(2026, 7, 7, tzinfo=UTC)

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        if "INSERT INTO chat_user" in query:
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

        if "WITH updated AS" in query:
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
            current_message["role"] = role
            current_message["content"] = content
            current_message["response_id"] = response_id
            current_message["metadata"] = metadata_json
            current_message["updated_at"] = self.now
            return current_message

        msg = f"Unhandled fetchrow query: {query}"
        raise AssertionError(msg)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
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
