import json
from uuid import uuid4

import pytest

from app.data.connection import Database
from app.llms import recommendation_tools
from app.llms.recommendation_tools import build_recommendation_tools
from app.llms.tools import agent_request_tools, get_agent_tools
from app.schemas.recommendations import (
    PersonScoreSchema,
    RecommendationEvidenceSchema,
    RecommendationResponseSchema,
)


def _response() -> RecommendationResponseSchema:
    person = PersonScoreSchema(
        name="Active Professor",
        score=0.9,
        defense_score=0.7,
        expertise_score=0.2,
    )
    return RecommendationResponseSchema(
        mode="full",
        mentor=person,
        mentor_is_given=False,
        members=[person, person],
        supporting_diploma_ids=[uuid4()],
        confidence=0.8,
        confidence_reasons=["strong_topic_match"],
        evidence=RecommendationEvidenceSchema(),
        alternatives=[],
    )


@pytest.mark.anyio
async def test_recommendation_tool_invokes_internal_service(monkeypatch) -> None:
    captured = {}
    db = Database("postgresql://test")

    async def fake_generate_committee_recommendation(payload, actual_db):
        captured["title"] = payload.title
        captured["db"] = actual_db
        return _response()

    monkeypatch.setattr(
        recommendation_tools,
        "generate_committee_recommendation",
        fake_generate_committee_recommendation,
    )

    tool = build_recommendation_tools(db)[0]

    result = await tool.ainvoke({"title": "AI систем за дипломска"})

    assert captured == {"title": "AI систем за дипломска", "db": db}
    assert json.loads(result)["mentor"]["name"] == "Active Professor"


@pytest.mark.anyio
async def test_agent_tools_include_request_scoped_recommendation_tool(
    monkeypatch,
) -> None:
    async def fake_get_mcp_tools():
        return []

    monkeypatch.setattr("app.llms.tools.get_mcp_tools", fake_get_mcp_tools)
    tool = build_recommendation_tools(Database("postgresql://test"))[0]

    with agent_request_tools([tool]):
        tools = await get_agent_tools()

    assert [item.name for item in tools] == ["recommend_diploma_committee"]
    assert await get_agent_tools() == []
