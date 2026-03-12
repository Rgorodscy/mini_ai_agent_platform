def execute_tool(tool_name: str, task: str) -> str:
    tool_name_lower = tool_name.lower()

    if "web-search" in tool_name_lower or "search" in tool_name_lower:
        return (
            f"[Mock Search Result] Found 3 relevant articles about: '{task}'. "
            f"Key findings: (1) Recent developments show significant progress."
            f"(2) Experts suggest further analysis is needed. "
            f"(3) Multiple sources confirm the trend."
        )

    if "summarizer" in tool_name_lower or "summary" in tool_name_lower:
        return (
            f"[Mock Summary] Summary of content related to '{task}': "
            f"The main topic covers key aspects with notable implications. "
            f"Three core points were identified and analyzed."
        )

    if "calculator" in tool_name_lower or "math" in tool_name_lower:
        return f"[Mock Calculator] Computed result for '{task}': 42"

    # Generic fallback
    return f"[Mock Tool: {tool_name}] Executed successfully for task: '{task}'"
