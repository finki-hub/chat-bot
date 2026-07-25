from re import Pattern, compile
from typing import Final

_UNRESOLVED_DISCORD_TOKEN: Final[Pattern[str]] = compile(
    r"<(?:@!?|@&|#)\d+>",
)
_UNRESOLVED_DISCORD_LABEL: Final = "[непозната Discord-ознака]"


def contains_unresolved_discord_token(value: str) -> bool:
    return _UNRESOLVED_DISCORD_TOKEN.search(value) is not None


def redact_unresolved_discord_tokens(value: str) -> str:
    return _UNRESOLVED_DISCORD_TOKEN.sub(_UNRESOLVED_DISCORD_LABEL, value)
