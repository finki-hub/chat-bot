import json
from uuid import uuid4

from tests.feedback_fake import FakeFeedbackDatabase


class DiscordFeedbackDatabase(FakeFeedbackDatabase):
    """Stateful SQL-boundary fake for HTTP retry -> real history-helper sequences."""

    async def fetchrow(self, query, *args):
        if "WHERE provider = 'discord' AND provider_subject = $1" in query:
            account = self.users.get(("discord", args[0]))
            return dict(account) if account else None
        if (
            "SELECT id FROM chat_conversation WHERE id = $1 AND user_id = $2 FOR UPDATE"
            in query
        ):
            return self._owned_conversation(*args)
        if "SELECT content, metadata->>'inferenceModel'" in query:
            message = self.messages.get(args[0])
            if (
                not message
                or message["conversation_id"] != args[1]
                or message["response_id"] != args[2]
            ):
                return None
            metadata = self._metadata(message)
            return {
                "content": message["content"],
                "inference_model": metadata.get("inferenceModel"),
                "embeddings_model": None,
                "query_transform_model": None,
            }
        if "INSERT INTO feedback" in query:
            key = args[:3]
            row = {"id": uuid4(), "response_id": args[0], "feedback_type": args[3]}
            self.feedback[key] = row
            return row
        return await super().fetchrow(query, *args)

    async def fetch(self, query, *args):
        if "owner.history_generation AS owner_generation" in query:
            found = []
            for message in self.messages.values():
                if (
                    message["response_id"] != args[0]
                    or message["role"] != "assistant"
                    or self._metadata(message).get("interface") != "discord"
                ):
                    continue
                parent = self.conversations[message["conversation_id"]]
                owner = next(
                    account
                    for account in self.users.values()
                    if account["id"] == parent["user_id"]
                )
                found.append(
                    {
                        "id": message["id"],
                        "conversation_id": parent["id"],
                        "owner_id": owner["id"],
                        "owner_generation": owner["history_generation"],
                    }
                )
            return found
        if "SELECT id, history_generation FROM chat_user WHERE id = ANY" in query:
            return [
                dict(account)
                for account in self.users.values()
                if account["id"] in args[0]
            ]
        return await super().fetch(query, *args)

    async def fetchval(self, query, *args, **kwargs):
        if "SELECT question.content FROM chat_message AS question" in query:
            answer = self.messages[args[0]]
            return next(
                row["content"]
                for row in self.messages.values()
                if row["conversation_id"] == answer["conversation_id"]
                and row["role"] == "user"
            )
        return await super().fetchval(query, *args, **kwargs)

    def _metadata(self, message):
        metadata = message["metadata"]
        return json.loads(metadata) if isinstance(metadata, str) else metadata
