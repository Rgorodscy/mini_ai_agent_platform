from dotenv import load_dotenv
import os

load_dotenv()


def _require(key: str) -> str:
    """Raises an error at startup if a required env variable is missing."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


DATABASE_URL = _require("DATABASE_URL")
MAX_EXECUTION_STEPS = int(os.getenv("MAX_EXECUTION_STEPS", "5"))
API_KEYS: dict[str, str] = {
    _require("API_KEY_TENANT_A"): "tenant_a",
    _require("API_KEY_TENANT_B"): "tenant_b",
    _require("API_KEY_TENANT_C"): "tenant_c",
}

SUPPORTED_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
