from app.models.agent import Agent
from app.core.mock_llm import call_mock_llm
from app.core.mock_tools import execute_tool
from app.config import MAX_EXECUTION_STEPS
from app.logger import get_logger

logger = get_logger(__name__)


def run_execution_loop(prompt: dict, agent: Agent) -> dict:
    """
    Runs the multi-step agent execution loop.
    - Calls mock LLM
    - If tool call requested: validates, executes, appends to context
    - Repeats until final response or max steps reached
    Returns a dict with steps taken and the final response.
    """
    available_tool_names = [tool.name for tool in agent.tools]
    context = []
    steps = []

    logger.info(f"Execution loop started | agent={agent.name} "
                f"tools={available_tool_names}")

    for step_num in range(1, MAX_EXECUTION_STEPS + 1):
        logger.info(f"Final response received | step={step_num}")
        llm_response = call_mock_llm(prompt, context, available_tool_names)

        if llm_response["type"] == "final_response":
            steps.append({
                "step": step_num,
                "type": "final_response",
                "content": llm_response["content"]
            })
            return {
                "steps": steps,
                "final_response": llm_response["content"],
                "status": "completed"
            }

        elif llm_response["type"] == "tool_call":
            tool_name = llm_response["tool"]

            if tool_name not in available_tool_names:
                logger.warning(f"Tool not assigned to agent | tool={tool_name}"
                               f" agent={agent.name}")
                steps.append({
                    "step": step_num,
                    "type": "tool_error",
                    "tool": tool_name,
                    "error": (
                        f"Tool '{tool_name}' is not assigned to this agent")
                })
                continue

            logger.info(f"Tool call | step={step_num} tool={tool_name}")
            tool_result = execute_tool(tool_name, llm_response["input"])

            step = {
                "step": step_num,
                "type": "tool_result",
                "tool": tool_name,
                "input": llm_response["input"],
                "result": tool_result
            }
            steps.append(step)
            context.append(step)

    logger.warning(f"Max steps reached | agent={agent.name} "
                   f"steps={MAX_EXECUTION_STEPS}")
    return {
        "steps": steps,
        "final_response": None,
        "status": "max_steps_reached",
    }
