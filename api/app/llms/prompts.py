from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "resources" / "prompts"


def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


DEFAULT_AGENT_SYSTEM_PROMPT = _load_prompt("agent_system.txt")
HYDE_SYSTEM_PROMPT = _load_prompt("hyde_system.txt")
DEFAULT_QUERY_TRANSFORM_SYSTEM_PROMPT = _load_prompt("query_transform_system.txt")


def build_user_agent_prompt(context: str, prompt: str) -> str:
    """
    Build a user prompt for agents with the context and user question.
    """
    return f"""Контекст од базата на знаења:
{context}

Прашање на корисникот: {prompt}"""


def build_agent_messages(
    system_prompt: str,
    history: list[BaseMessage],
    user_prompt: str,
) -> list[BaseMessage]:
    """
    Assemble the message list for a chat turn:
    the system prompt, any prior conversation turns, then the latest user prompt.
    """
    return [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=user_prompt),
    ]


def stitch_system_user(system: str, user_prompt: str) -> str:
    """
    Stitch the system prompt and user prompt into a single string for the LLM.
    """
    return f"<|system|> {system}\n\n<|user|> {user_prompt}\n\n<|assistant|>"


def stitch_conversation(
    system: str,
    history: list[BaseMessage],
    user_prompt: str,
) -> str:
    """
    Stitch the system prompt, prior conversation turns and the latest user prompt
    into a single tagged string for models that take a plain text prompt.
    """
    parts = [f"<|system|> {system}"]
    for message in history:
        tag = "user" if isinstance(message, HumanMessage) else "assistant"
        parts.append(f"<|{tag}|> {message.content}")
    parts.append(f"<|user|> {user_prompt}")
    parts.append("<|assistant|>")
    return "\n\n".join(parts)


def history_transcript(history: list[BaseMessage]) -> str:
    """
    Render prior conversation turns as a plain readable transcript, for models
    that only accept a single prompt string (e.g. the GPU API).
    """
    lines = []
    for message in history:
        label = "Корисник" if isinstance(message, HumanMessage) else "Асистент"
        lines.append(f"{label}: {message.content}")
    return "\n".join(lines)


def to_history_messages(
    history: list[tuple[str, str]],
) -> list[BaseMessage]:
    """
    Convert (role, content) tuples into LangChain messages.
    'user' becomes a HumanMessage; anything else becomes an AIMessage.
    """
    return [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in history
    ]
