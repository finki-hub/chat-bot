from pathlib import Path

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


def stitch_system_user(system: str, user_prompt: str) -> str:
    """
    Stitch the system prompt and user prompt into a single string for the LLM.
    """
    return f"<|system|> {system}\n\n<|user|> {user_prompt}\n\n<|assistant|>"
