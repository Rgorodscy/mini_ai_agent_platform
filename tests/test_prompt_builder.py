from unittest.mock import MagicMock
from app.core.prompt_builder import build_prompt


def make_mock_agent(name="Test Agent",
                    role="researcher",
                    description="A test agent",
                    tools=None):
    agent = MagicMock()
    agent.name = name
    agent.role = role
    agent.description = description
    agent.tools = tools or []
    return agent


def make_mock_tool(name="web-search", description="Searches the web"):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


def test_prompt_has_required_keys():
    agent = make_mock_agent()
    result = build_prompt(agent, "test task")
    assert isinstance(result, dict)
    assert "system" in result
    assert "tools" in result
    assert "user" in result


def test_prompt_user_matches_task():
    agent = make_mock_agent()
    result = build_prompt(agent, "summarize the Q2 report")
    assert result["user"] == "summarize the Q2 report"


def test_prompt_system_contains_agent_name():
    agent = make_mock_agent(name="Research Agent")
    result = build_prompt(agent, "some task")
    assert "Research Agent" in result["system"]


def test_prompt_system_contains_agent_role():
    agent = make_mock_agent(role="data analyst")
    result = build_prompt(agent, "some task")
    assert "data analyst" in result["system"]


def test_prompt_system_contains_agent_description():
    agent = make_mock_agent(description="Analyzes financial data")
    result = build_prompt(agent, "some task")
    assert "Analyzes financial data" in result["system"]


def test_prompt_tools_empty_when_no_tools():
    agent = make_mock_agent(tools=[])
    result = build_prompt(agent, "some task")
    assert result["tools"] == []


def test_prompt_tools_contains_tool_name():
    tool = make_mock_tool(name="web-search")
    agent = make_mock_agent(tools=[tool])
    result = build_prompt(agent, "some task")
    assert result["tools"][0]["name"] == "web-search"


def test_prompt_tools_contains_tool_description():
    tool = make_mock_tool(description="Searches the web for information")
    agent = make_mock_agent(tools=[tool])
    result = build_prompt(agent, "some task")
    assert result["tools"][0]["description"] == (
        "Searches the web for information")


def test_prompt_multiple_tools():
    tools = [
        make_mock_tool(name="web-search", description="Searches the web"),
        make_mock_tool(name="summarizer", description="Summarizes text"),
    ]
    agent = make_mock_agent(tools=tools)
    result = build_prompt(agent, "some task")
    assert len(result["tools"]) == 2
    tool_names = [t["name"] for t in result["tools"]]
    assert "web-search" in tool_names
    assert "summarizer" in tool_names
