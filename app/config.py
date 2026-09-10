import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Raises an error if a required env variable is missing."""
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

# --- LLM / RAG settings ---

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

SUPPORTED_MODELS = [GROQ_MODEL]


@lru_cache(maxsize=1)
def get_groq_client():
    """
    Lazily builds the Groq client.

    Built on first use rather than at import time so that importing the
    application never requires network access or an API key — tests stub
    this out entirely.
    """
    from groq import Groq

    return Groq(api_key=_require("GROQ_API_KEY"))


def validate_settings() -> None:
    """
    Fails fast on missing configuration at application startup.

    Import-time validation is deliberately limited to the settings above;
    anything that would reach out to the network (LLM client, embedding
    models) is validated here instead, so the package stays importable in
    test and tooling contexts.
    """
    _require("GROQ_API_KEY")
