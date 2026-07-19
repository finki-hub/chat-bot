import json
from uuid import uuid4

from tests.chat_persistence_fake import FakeChatDatabase


class FakeFeedbackDatabase(FakeChatDatabase):
    def __init__(self) -> None:
        super().__init__()
        self.fetchrow_calls = 0

    async def fetchrow(
        self,
        query: str,
        *args: object,
    ) -> dict[str, object] | None:
        self.fetchrow_calls += 1
        if "WITH owned_response AS" in query and "INSERT INTO feedback" in query:
            (
                response_id,
                user_id,
                feedback_type,
                client_ref,
                channel_id,
                guild_id,
                embeddings_model,
                query_transform_model,
            ) = args
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
                parsed = json.loads(metadata) if isinstance(metadata, str) else metadata
                if not isinstance(parsed, dict):
                    parsed = {}
                inference_model = parsed.get("inferenceModel")
                parsed["feedback"] = feedback_type
                assistant["metadata"] = parsed
                assistant["updated_at"] = self.now
                feedback_key = (response_id, "web", user_id)
                current = self.feedback.get(feedback_key)
                if current is None:
                    current = {
                        "answer_text": assistant["content"],
                        "channel_id": channel_id,
                        "client": "web",
                        "client_ref": client_ref,
                        "created_at": self.now,
                        "embeddings_model": embeddings_model,
                        "feedback_type": feedback_type,
                        "guild_id": guild_id,
                        "id": uuid4(),
                        "inference_model": inference_model,
                        "query_transform_model": query_transform_model,
                        "question_text": None
                        if question is None
                        else question["content"],
                        "response_id": response_id,
                        "updated_at": self.now,
                        "user_id": user_id,
                    }
                    self.feedback[feedback_key] = current
                else:
                    current["feedback_type"] = feedback_type
                    current["updated_at"] = self.now
                return current
            return None

        if "DELETE FROM feedback AS stored" not in query:
            return await super().fetchrow(query, *args)

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
            self.feedback.pop((response_id, "web", user_id), None)
            metadata = assistant["metadata"]
            parsed = json.loads(metadata) if isinstance(metadata, str) else metadata
            if not isinstance(parsed, dict):
                parsed = {}
            parsed.pop("feedback", None)
            assistant["metadata"] = parsed
            assistant["updated_at"] = self.now
            return {"id": assistant["id"]}
        return None
