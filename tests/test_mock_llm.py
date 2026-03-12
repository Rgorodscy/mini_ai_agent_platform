from app.core.mock_llm import call_mock_llm


def make_prompt(task: str, system: str = "You are a Test Agent, researcher."):
    return {"system": system, "tools": [], "user": task}


def test_returns_final_response_structure():
    prompt = make_prompt("hello")
    result = call_mock_llm(prompt, [], [])
    assert "type" in result
    assert result["type"] == "final_response"
    assert "content" in result


def test_returns_tool_call_structure():
    prompt = make_prompt("search for AI trends")
    result = call_mock_llm(prompt, [], ["web-search"])
    assert "type" in result
    assert result["type"] == "tool_call"
    assert "tool" in result
    assert "input" in result


def test_search_keyword_triggers_web_search():
    prompt = make_prompt("search for the latest AI news")
    result = call_mock_llm(prompt, [], ["web-search"])
    assert result["type"] == "tool_call"
    assert result["tool"] == "web-search"


def test_find_keyword_triggers_web_search():
    prompt = make_prompt("find information about climate change")
    result = call_mock_llm(prompt, [], ["web-search"])
    assert result["type"] == "tool_call"
    assert result["tool"] == "web-search"


def test_summarize_keyword_triggers_summarizer():
    prompt = make_prompt("summarize this document")
    result = call_mock_llm(prompt, [], ["summarizer"])
    assert result["type"] == "tool_call"
    assert result["tool"] == "summarizer"


def test_calculate_keyword_triggers_calculator():
    prompt = make_prompt("calculate the total revenue")
    result = call_mock_llm(prompt, [], ["calculator"])
    assert result["type"] == "tool_call"
    assert result["tool"] == "calculator"


def test_no_tool_available_returns_final_response():
    prompt = make_prompt("search for AI news")
    result = call_mock_llm(prompt, [], [])  # no tools available
    assert result["type"] == "final_response"


def test_tool_not_in_available_list_returns_final_response():
    prompt = make_prompt("search for AI news")
    result = call_mock_llm(prompt, [], ["summarizer"])
    # web-search not available
    assert result["type"] == "final_response"


def test_already_used_tool_not_called_again():
    prompt = make_prompt("search for AI news")
    context = [{"type": "tool_result", "tool": "web-search",
                "result": "some result"}]
    result = call_mock_llm(prompt, context, ["web-search"])
    # web-search already used, should return final response
    assert result["type"] == "final_response"


def test_final_response_includes_tool_results():
    prompt = make_prompt("search for AI news")
    context = [{"type": "tool_result", "tool": "web-search",
                "result": "mock result data"}]
    result = call_mock_llm(prompt, context, ["web-search"])
    assert result["type"] == "final_response"
    assert "mock result data" in result["content"]


def test_same_input_returns_same_output():
    prompt = make_prompt("search for AI news")
    result1 = call_mock_llm(prompt, [], ["web-search"])
    result2 = call_mock_llm(prompt, [], ["web-search"])
    assert result1 == result2
