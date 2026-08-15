import os

import anyio
import pytest

from app.data.questions import get_matching_questions
from tests.integration.embedding_postgres_test_support import database

pytestmark = pytest.mark.skipif(
    os.getenv("TEST_DATABASE_URL") is None,
    reason="set TEST_DATABASE_URL to run real-PostgreSQL lexical FAQ tests",
)


def test_lexical_faq_search_uses_safe_or_matching_and_title_weighting() -> None:
    async def run() -> None:
        async with database() as current_database:
            await current_database.execute(
                """
                INSERT INTO question (id, name, content) VALUES
                    ('00000000-0000-0000-0000-000000000401', 'Alpha policy', 'unrelated text'),
                    ('00000000-0000-0000-0000-000000000402', 'Beta details', 'alpha appears in content'),
                    ('00000000-0000-0000-0000-000000000403', 'Студиски програми', 'Информации за достапните насоки.')
                """,
            )

            weighted = await get_matching_questions(
                current_database,
                "alpha absent",
                limit=2,
            )
            cyrillic = await get_matching_questions(
                current_database,
                "студиски програми",
                limit=1,
            )
            punctuation = await get_matching_questions(
                current_database,
                "https://example.com/?ad=qwe&",
                limit=2,
            )

        assert [question.name for question in weighted] == [
            "Alpha policy",
            "Beta details",
        ]
        assert [question.name for question in cyrillic] == ["Студиски програми"]
        assert len(punctuation) <= 2

    anyio.run(run)
