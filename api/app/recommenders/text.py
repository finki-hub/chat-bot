def build_proposal_text(title: str, description: str | None) -> str:
    title = title.strip()
    description = (description or "").strip()
    return f"{title}\n{description}" if description else title
