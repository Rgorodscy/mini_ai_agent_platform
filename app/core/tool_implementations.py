import requests

from app.logger import get_logger

logger = get_logger(__name__)


def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"ERROR: invalid expression — {e}"


def search_knowledge(query: str) -> str:
    # Placeholder
    logger.info(f"search_knowledge called | query={query}")
    return f"No results found for: '{query}' (RAG not implemented yet)"


def summarize(text: str) -> str:
    # Placeholder
    words = text.split()
    if len(words) <= 30:
        return text
    return " ".join(words[:30]) + "..."


def get_weather(city: str) -> str:
    try:
        lat, long = get_coordinates(city)
    except ValueError as e:
        return str(e)
    temp_res = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&current=temperature_2m,weathercode"
    )
    temp_res_json = temp_res.json()
    temperature = temp_res_json.get("current").get("temperature_2m")
    weathercode = temp_res_json.get("current").get("weathercode")

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


def get_coordinates(city: str):
    res = requests.get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en"
    )
    res_json = res.json()
    results = res_json.get("results")
    if not results:
        raise ValueError(f"City '{city}' not found.")
    res_lat = results[0].get("latitude")
    res_long = results[0].get("longitude")
    return res_lat, res_long


def web_search(query: str) -> str:
    # Placeholder
    logger.info(f"web_search called | query={query}")
    return f"Simulated results to: '{query}'"


def think(reasoning: str) -> str:
    return "ok"


# --- Registry ---

TOOL_REGISTRY: dict[str, dict] = {
    "calculator": {
        "func": calculator,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '10 * 3 + 5'",
                }
            },
            "required": ["expression"],
        },
    },
    "search_knowledge": {
        "func": search_knowledge,
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
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"}
            },
            "required": ["text"],
        },
    },
    "think": {
        "func": think,
        "description": "Reason step by step before taking an action. Does not affect the external world.",
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
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}
