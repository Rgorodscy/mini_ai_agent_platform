import json
import operator
from typing import TypedDict, Annotated, Any
from langgraph.graph import StateGraph, END

from app.models.agent import Agent
from app.core.tool_implementations import TOOL_REGISTRY
from app.config import MAX_EXECUTION_STEPS, groq_client
from app.logger import get_logger

logger = get_logger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"


class ExecutionState(TypedDict):
    messages: Annotated[list, operator.add]
    steps: Annotated[list, operator.add]
    active_tools: list
    tool_map: dict[str, Any]
    consecutive_errors: int
    force_text: bool
    max_steps: int


def _build_tools_for_agent(
    agent: Agent, tenant_id: str
) -> tuple[list[dict], dict]:
    """
    Cross checks the tools in the agent's database with
    the available implementations in the TOOL_REGISTRY.
    Returns (schemas, tool_map).
    """
    schemas = []
    tool_map = {}

    for tool in agent.tools:
        impl = TOOL_REGISTRY.get(tool.name)
        if not impl:
            logger.warning(
                f"Non implemented tool — ignored | tool={tool.name} agent={agent.name}"
            )
            continue

        description = (
            tool.description
            if isinstance(tool.description, str)
            else f"Tool: {tool.name}"
        )

        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": description,
                    "parameters": impl["parameters"],
                },
            }
        )
        # inject tenant_id to search_knowledge
        if tool.name == "search_knowledge":
            from app.core.tool_implementations import make_search_knowledge

            tool_map[tool.name] = make_search_knowledge(tenant_id)
        else:
            tool_map[tool.name] = impl["func"]

    return schemas, tool_map


def _current_step(state: ExecutionState) -> int:
    return len(state["steps"]) + 1


# --- Graph nodes ---
def call_model(state: ExecutionState) -> dict:
    tool_choice = "none" if state.get("force_text") else "auto"

    def _call(choice):
        return groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=state["messages"],
            tools=state["active_tools"],
            tool_choice=choice,
        )

    try:
        response = _call(tool_choice)
        return {"messages": [response.choices[0].message], "force_text": False}
    except Exception as e:
        err = str(e)
        if "400" in err and "tool_use_failed" in err:
            logger.warning("Invalid tool format — retry.")
            try:
                response = _call("auto")
                return {
                    "messages": [response.choices[0].message],
                    "force_text": False,
                }
            except Exception:
                logger.warning("Retry failed — responding without tools.")
                response = _call("none")
                return {
                    "messages": [response.choices[0].message],
                    "force_text": False,
                }
        raise


def execute_tools(state: ExecutionState) -> dict:
    last_message = state["messages"][-1]
    tool_results = []
    new_steps = []
    consecutive_errors = state.get("consecutive_errors", 0)
    force_text = False
    step_num = _current_step(state)
    tool_map = state["tool_map"]

    for tool_call in last_message.tool_calls:
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            args = {}

        # think doesn't generate step in execution history
        if name == "think":
            logger.info(f"Think | reasoning={args.get('reasoning', '')[:80]}")
            tool_results.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": "ok"}
            )
            continue

        # tool doesn't exist or is not available in agent
        func = tool_map.get(name)
        if not func:
            error = f"Tool '{name}' not available for the agent."
            logger.warning(f"Unavailable tool | tool={name}")
            new_steps.append(
                {
                    "step": step_num,
                    "type": "tool_error",
                    "tool": name,
                    "error": error,
                }
            )
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"ERROR: {error}",
                }
            )
            consecutive_errors += 1
            force_text = True
            step_num += 1
            continue

        try:
            result = func(**args)
        except TypeError as e:
            result = f"ERROR: invalid arguments — {e}"
        except Exception as e:
            result = f"ERROR unexpected: {e}"

        if result.startswith("ERROR"):
            consecutive_errors += 1
            force_text = True
            new_steps.append(
                {
                    "step": step_num,
                    "type": "tool_error",
                    "tool": name,
                    "input": args,
                    "error": result,
                }
            )
            logger.warning(f"Tool error | tool={name} error={result}")
        else:
            consecutive_errors = 0
            new_steps.append(
                {
                    "step": step_num,
                    "type": "tool_result",
                    "tool": name,
                    "input": args,
                    "result": result,
                }
            )
            logger.info(f"Tool success | tool={name} step={step_num}")

        tool_results.append(
            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
        )
        step_num += 1

    return {
        "messages": tool_results,
        "steps": new_steps,
        "consecutive_errors": consecutive_errors,
        "force_text": force_text,
    }


# --- Conditional edges ---


def should_continue(state: ExecutionState) -> str:
    if state.get("consecutive_errors", 0) >= 3:
        return END

    if len(state["steps"]) >= state["max_steps"]:
        return END

    last = state["messages"][-1]
    if hasattr(last, "role") and last.role == "assistant":
        if not last.tool_calls:
            return END

    return "execute_tools"


def after_tools(_: ExecutionState) -> str:
    return "call_model"


# --- Graph compilation ---


def _build_graph():
    graph = StateGraph(ExecutionState)
    graph.add_node("call_model", call_model)
    graph.add_node("execute_tools", execute_tools)
    graph.set_entry_point("call_model")
    graph.add_conditional_edges("call_model", should_continue)
    graph.add_conditional_edges("execute_tools", after_tools)
    return graph.compile()


_agent_graph = _build_graph()


def run_execution_loop(
    prompt: dict, agent: Agent, tenant_id: str = ""
) -> dict:
    """
    Runs the multi-step agent execution loop.
    - Calls LLM
    - If tool call requested: validates, executes, appends to context
    - Repeats until final response or max steps reached
    Returns a dict with steps taken and the final response.
    """
    active_tools, tool_map = _build_tools_for_agent(agent, tenant_id)

    if not active_tools:
        logger.warning(f"Agent without implemented tools | agent={agent.name}")

    all_tools = active_tools

    system_content = (
        f"You are {agent.name}. {agent.role}.\n{agent.description}\n\n"
        f"Use the available tools when needed. Included one that searches "
        f"for internal knowledge, called search_knowledge if needed. "
        f"Use it if you think it's necessary"
        f" to find information to complete the task. Think before acting. "
        f"When you have enough information, respond directly without calling"
        f" any tool.If there is information you don't know, "
        f"say you don't know, don't try to guess or make it up.\n\n"
    )

    initial_state: ExecutionState = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt.get("user", "")},
        ],
        "steps": [],
        "active_tools": all_tools,
        "tool_map": tool_map,
        "consecutive_errors": 0,
        "force_text": False,
        "max_steps": MAX_EXECUTION_STEPS,
    }

    logger.info(
        f"LangGraph execution started | agent={agent.name} "
        f"tools={[t['function']['name'] for t in active_tools]}"
    )

    final_state = _agent_graph.invoke(initial_state)

    # Extracts final response from the assistants last message
    final_response = None
    for msg in reversed(final_state["messages"]):
        if hasattr(msg, "role") and msg.role == "assistant" and msg.content:
            final_response = msg.content
            break

    steps = final_state["steps"]

    if final_response:
        steps = steps + [
            {
                "step": len(steps) + 1,
                "type": "final_response",
                "content": final_response,
            }
        ]

    status = "completed" if final_response else "max_steps_reached"

    logger.info(
        f"LangGraph execution done | agent={agent.name} "
        f"status={status} steps={len(steps)}"
    )

    return {"steps": steps, "final_response": final_response, "status": status}
