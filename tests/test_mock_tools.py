from app.core.mock_tools import execute_tool


def test_web_search_returns_result():
    result = execute_tool("web-search", "AI trends")
    assert result is not None
    assert len(result) > 0


def test_web_search_contains_mock_indicator():
    result = execute_tool("web-search", "AI trends")
    assert "Mock" in result


def test_web_search_is_deterministic():
    result1 = execute_tool("web-search", "AI trends")
    result2 = execute_tool("web-search", "AI trends")
    assert result1 == result2


def test_search_variant_name():
    result = execute_tool("search-tool", "some query")
    assert "Mock" in result


def test_summarizer_returns_result():
    result = execute_tool("summarizer", "long document text")
    assert result is not None
    assert len(result) > 0


def test_summarizer_is_deterministic():
    result1 = execute_tool("summarizer", "some text")
    result2 = execute_tool("summarizer", "some text")
    assert result1 == result2


def test_summary_variant_name():
    result = execute_tool("summary-tool", "some text")
    assert "Mock" in result


def test_calculator_returns_result():
    result = execute_tool("calculator", "2 + 2")
    assert result is not None
    assert len(result) > 0


def test_calculator_is_deterministic():
    result1 = execute_tool("calculator", "2 + 2")
    result2 = execute_tool("calculator", "2 + 2")
    assert result1 == result2


def test_math_variant_name():
    result = execute_tool("math-tool", "some equation")
    assert "Mock" in result


def test_unknown_tool_returns_generic_result():
    result = execute_tool("unknown-tool", "some task")
    assert "Mock Tool" in result
    assert "unknown-tool" in result


def test_unknown_tool_is_deterministic():
    result1 = execute_tool("unknown-tool", "some task")
    result2 = execute_tool("unknown-tool", "some task")
    assert result1 == result2
