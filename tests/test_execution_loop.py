from unittest.mock import MagicMock
from app.core.execution_loop import run_execution_loop


def make_mock_agent(tool_names: list[str] = None):
    agent = MagicMock()
    agent.name = "Test Agent"
    agent.tools = []

    for name in (tool_names or []):
        tool = MagicMock()
        tool.name = name
        agent.tools.append(tool)

    return agent


def make_prompt(task: str):
    return {
        "system": "You are Test Agent, researcher.",
        "tools": [],
        "user": task
    }


def test_returns_required_keys():
    agent = make_mock_agent()
    prompt = make_prompt("hello")
    result = run_execution_loop(prompt, agent)
    assert "steps" in result
    assert "final_response" in result
    assert "status" in result


def test_completed_status_on_success():
    agent = make_mock_agent()
    prompt = make_prompt("hello")
    result = run_execution_loop(prompt, agent)
    assert result["status"] == "completed"


def test_final_response_is_string_on_success():
    agent = make_mock_agent()
    prompt = make_prompt("hello")
    result = run_execution_loop(prompt, agent)
    assert isinstance(result["final_response"], str)
    assert len(result["final_response"]) > 0


def test_steps_is_list():
    agent = make_mock_agent()
    prompt = make_prompt("hello")
    result = run_execution_loop(prompt, agent)
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) > 0


def test_tool_call_step_recorded():
    agent = make_mock_agent(tool_names=["web-search"])
    prompt = make_prompt("search for AI trends")
    result = run_execution_loop(prompt, agent)
    step_types = [s["type"] for s in result["steps"]]
    assert "tool_result" in step_types


def test_final_response_step_recorded():
    agent = make_mock_agent()
    prompt = make_prompt("hello")
    result = run_execution_loop(prompt, agent)
    step_types = [s["type"] for s in result["steps"]]
    assert "final_response" in step_types


def test_tool_not_assigned_to_agent_is_rejected():
    agent = make_mock_agent(tool_names=[])
    prompt = make_prompt("search for AI trends")
    result = run_execution_loop(prompt, agent)
    assert result["status"] == "completed"


def test_max_steps_safeguard(monkeypatch):
    """Force the mock LLM to always return tool calls to trigger max steps."""
    from app.core import execution_loop

    def always_tool_call(prompt, context, available_tool_names):
        return {"type": "tool_call", "tool": "web-search", "input": "test"}

    monkeypatch.setattr(execution_loop, "call_mock_llm", always_tool_call)

    agent = make_mock_agent(tool_names=["web-search"])
    prompt = make_prompt("search forever")
    result = run_execution_loop(prompt, agent)

    assert result["status"] == "max_steps_reached"
    assert result["final_response"] is None


def test_max_steps_does_not_exceed_limit(monkeypatch):
    from app.core import execution_loop
    from app.config import MAX_EXECUTION_STEPS

    def always_tool_call(prompt, context, available_tool_names):
        return {"type": "tool_call", "tool": "web-search", "input": "test"}

    monkeypatch.setattr(execution_loop, "call_mock_llm", always_tool_call)

    agent = make_mock_agent(tool_names=["web-search"])
    prompt = make_prompt("search forever")
    result = run_execution_loop(prompt, agent)

    tool_steps = [s for s in result["steps"] if s["type"] == "tool_result"]
    assert len(tool_steps) <= MAX_EXECUTION_STEPS


def test_full_flow_with_tool(monkeypatch):
    """Verify tool call → result → final response flow."""
    from app.core import execution_loop

    call_count = {"n": 0}

    def controlled_llm(prompt, context, available_tool_names):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"type": "tool_call", "tool": "web-search",
                    "input": "AI news"}
        return {"type": "final_response", "content": "Done!"}

    monkeypatch.setattr(execution_loop, "call_mock_llm", controlled_llm)

    agent = make_mock_agent(tool_names=["web-search"])
    prompt = make_prompt("search for AI news")
    result = run_execution_loop(prompt, agent)

    assert result["status"] == "completed"
    assert result["final_response"] == "Done!"
    step_types = [s["type"] for s in result["steps"]]
    assert "tool_result" in step_types
    assert "final_response" in step_types
