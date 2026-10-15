from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import HttpUrl

from app.llms.context import _chunk_candidate, _question_candidate
from app.schemas.documents import ChunkSchema
from app.schemas.questions import QuestionSchema
from tests.eval.answer_eval import AnswerCase, load_answer_cases

_CASES = tuple(
    case
    for case in load_answer_cases(Path(__file__).with_name("answer_golden.jsonl"))
    if case.query
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_grounded_context_blocks_match_production_formatters(case: AnswerCase):
    blocks = case.context.split("\n\nТип на извор: ")
    for index, value in enumerate(blocks):
        block = value if index == 0 else f"Тип на извор: {value}"
        header, separator, content = block.partition("\nСодржина: ")
        assert separator, case.id
        fields: dict[str, str] = {}
        for line in header.splitlines():
            label, separator, text = line.partition(": ")
            assert separator, (case.id, line)
            assert label not in fields, (case.id, line)
            fields[label] = text

        source_type = fields.pop("Тип на извор")
        if source_type == "Документ":
            title = fields.pop("Извор")
            section = fields.pop("Секција", None)
            if section:
                assert title.endswith(f" ({section})")
                title = title.removesuffix(f" ({section})")
            authority_url = fields.pop("Авторитетен URL", None)
            chunk = ChunkSchema(
                id=UUID(int=index + 1),
                document_id=UUID(int=1),
                document_name="frozen-answer-evaluation",
                document_title=title,
                document_authority_url=(
                    HttpUrl(authority_url) if authority_url else None
                ),
                document_current_status=fields.pop("Статус (од корпусот)", None),
                document_date=fields.pop("Датум на документот", None),
                document_last_verified=fields.pop("Последна проверка на изворот", None),
                section=section,
                chunk_index=index,
                content=content,
            )
            rendered = _chunk_candidate(chunk).context_text
        else:
            assert source_type == "FAQ", (case.id, source_type)
            question = QuestionSchema(
                id=UUID(int=index + 1),
                name=fields.pop("Наслов"),
                content=content,
                created_at=datetime(2026, 9, 14, tzinfo=UTC),
                updated_at=datetime(2026, 9, 14, tzinfo=UTC),
            )
            rendered = _question_candidate(question).context_text

        assert not fields, (case.id, fields)
        assert block == rendered, case.id
