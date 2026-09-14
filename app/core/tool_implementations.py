import requests

from app.core.safe_math import UnsafeExpression, safe_eval
from app.logger import get_logger

logger = get_logger(__name__)

HTTP_TIMEOUT = 10


def calculator(expression: str) -> str:
    """Evaluates an arithmetic expression. See core.safe_math."""
    try:
        return str(safe_eval(expression))
    except UnsafeExpression as e:
        logger.warning(
            f"Rejected expression | expression={expression[:80]!r} reason={e}"
        )
        return f"ERROR: invalid expression — {e}"
    except ZeroDivisionError:
        return "ERROR: invalid expression — division by zero"
    except (OverflowError, ValueError) as e:
        return f"ERROR: could not compute — {e}"


def make_search_knowledge(tenant_id: str):
    """
    Builds a search_knowledge bound to one tenant.

    The tenant is closed over rather than passed as a tool argument so the
    model has no way to name a collection it should not reach.
    """

    def search_knowledge(query: str) -> str:
        from app.core.rag import answer_from_knowledge
        from app.core.rag.pipeline import NO_RESULTS

        try:
            results = answer_from_knowledge(query, tenant_id)
            # The pipeline signals "nothing found" with NO_RESULTS, not an
            # empty string. Checking only for emptiness wrapped that message
            # in "Relevant knowledge for ...", handing the model a
            # contradiction: relevant knowledge, then no relevant document.
            if not results or results == NO_RESULTS:
                return f"No relevant documents found for: '{query}'"
            return f"Relevant knowledge for '{query}':\n\n{results}"
        except Exception as e:
            logger.error(
                f"Error in search_knowledge | query={query} error={e}"
            )
            return f"ERROR: failed to retrieve knowledge — {e}"

    return search_knowledge


def summarize(text: str) -> str:
    # Placeholder: truncates rather than summarizing.
    words = text.split()
    if len(words) <= 30:
        return text
    return " ".join(words[:30]) + "..."


def get_weather(city: str) -> str:
    try:
        lat, long = get_coordinates(city)
    except ValueError as e:
        return f"ERROR: {e}"
    except requests.RequestException as e:
        return f"ERROR: geocoding request failed — {e}"

    try:
        temp_res = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": long,
                "current": "temperature_2m,weathercode",
            },
            timeout=HTTP_TIMEOUT,
        )
        temp_res.raise_for_status()
        current = temp_res.json().get("current", {})
    except requests.RequestException as e:
        return f"ERROR: weather request failed — {e}"
    except ValueError as e:
        return f"ERROR: malformed weather response — {e}"

    temperature = current.get("temperature_2m")
    weathercode = current.get("weathercode")

    weather_descriptions = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        61: "light rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "light snow",
        80: "rain showers",
        95: "thunderstorm",
    }

    description = weather_descriptions.get(weathercode, f"code {weathercode}")
    return f"Temperature in {city}: {temperature}°C, condition: {description}"


def get_coordinates(city: str) -> tuple[float, float]:
    res = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en"},
        timeout=HTTP_TIMEOUT,
    )
    res.raise_for_status()
    results = res.json().get("results")
    if not results:
        raise ValueError(f"City '{city}' not found.")
    return results[0].get("latitude"), results[0].get("longitude")


def web_search(query: str) -> str:
    # Placeholder: returns a canned string, makes no request.
    logger.info(f"web_search called | query={query}")
    return f"Simulated results to: '{query}'"


def think(reasoning: str) -> str:
    return "ok"


# --- Registry ---
#
# A tool row in the database only becomes callable if its name appears
# here. The registry owns the JSON schema the model is shown; the database
# row owns the tenant-facing description.

TOOL_REGISTRY: dict[str, dict] = {
    "calculator": {
        "func": calculator,
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Arithmetic expression to evaluate, "
                        "e.g. '10 * 3 + 5'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
    "search_knowledge": {
        # Bound per tenant at execution time by make_search_knowledge.
        "func": None,
        "description": (
            "Search the tenant's internal knowledge base for information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up information",
                }
            },
            "required": ["query"],
        },
    },
    "summarize": {
        "func": summarize,
        "description": "Shorten a long piece of text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"}
            },
            "required": ["text"],
        },
    },
    "get_weather": {
        "func": get_weather,
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Lisbon'",
                }
            },
            "required": ["city"],
        },
    },
    "think": {
        "func": think,
        "description": (
            "Reason step by step before taking an action. "
            "Does not affect the external world."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your internal reasoning",
                }
            },
            "required": ["reasoning"],
        },
    },
    "web_search": {
        "func": web_search,
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}

AVAILABLE_TOOL_NAMES = sorted(TOOL_REGISTRY)
