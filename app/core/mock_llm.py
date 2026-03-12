TOOL_TRIGGER_KEYWORDS = {
    "web-search": ["search", "find", "look up", "research", "what is",
                   "who is", "latest"],
    "summarizer": ["summarize", "summary", "tldr", "condense", "shorten"],
    "calculator": ["calculate", "compute", "math", "sum", "multiply",
                   "divide"],
}


def call_mock_llm(
    prompt: str,
    context: list[dict],
    available_tool_names: list[str]
) -> dict:
    """
    Simulates an LLM response deterministically.
    Returns either:
      - {"type": "tool_call", "tool": <name>, "input": <str>}
      - {"type": "final_response", "content": <str>}

    Decision logic:
    - On first call (no tool results in context yet),
    check if task matches a tool keyword
    - If a matching tool is available, request that tool call
    - Otherwise (or on subsequent calls), return a final response
    """
    task = prompt["user"].lower()
    already_used_tools = {step["tool"] for step in
                          context if step.get("type") == "tool_result"}

    for tool_name in available_tool_names:
        if tool_name in already_used_tools:
            continue

        tool_key = _match_tool(tool_name)
        if tool_key and any(kw in task for kw in
                            TOOL_TRIGGER_KEYWORDS.get(tool_key, [])):
            return {
                "type": "tool_call",
                "tool": tool_name,
                "input": prompt["user"]
            }

    tool_results = [
        step["result"] for step in context if step.get("type") == "tool_result"
    ]

    if tool_results:
        combined_results = " | ".join(tool_results)
        content = (
            f"Based on the tool results: {combined_results}. "
            f"Final answer for '{prompt["user"]}': "
            f"The analysis is complete. "
            f"The gathered information confirms the key findings above"
        )
    else:
        content = (
            f"Mock response for task: '{prompt['user']}'. "
            f"As {prompt['system'].splitlines()[0]}, "
            f"I have completed the requested task successfully."
        )

    return {
        "type": "final_response",
        "content": content
    }


def _match_tool(tool_name: str) -> str | None:
    tool_name_lower = tool_name.lower()
    for key in TOOL_TRIGGER_KEYWORDS:
        if key in tool_name_lower:
            return key
    return None
