from app.models.agent import Agent


def build_prompt(agent: Agent, task: str) -> str:
    tool_descriptions = [
        {"name": tool.name, "description": tool.description} for
        tool in agent.tools
    ]

    system_instructions = (
        f"You are {agent.name}, {agent.role}. \n"
        f"{agent.description}\n"
        f"Your goal is to help the user by completing their task accurately \n"
        "and thoroughly.\n"
    )

    return {
        "system": system_instructions,
        "tools": tool_descriptions,
        "user": task
    }
