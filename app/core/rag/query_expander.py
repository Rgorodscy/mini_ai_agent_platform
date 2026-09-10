import json
import re

from app.config import GROQ_MODEL, get_groq_client
from app.logger import get_logger

logger = get_logger(__name__)

EXPAND_PROMPT = """
You are an assistant that expands search queries.
Return ONLY a JSON array of strings.
No additional text, no markdown, no explanations.
Example: ["original query", "variation 1", "variation 2", "variation 3"]
"""


def expand_query(query: str) -> list[str]:
    """
    Rewrites the query into several phrasings to widen retrieval recall.

    Always returns at least the original query: a malformed model response
    degrades recall but must never fail the search.
    """
    try:
        res = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": EXPAND_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        variations = _parse_variations(res.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Query expansion failed | error={e}")
        return [query]

    if query not in variations:
        variations.insert(0, query)

    return variations


def _parse_variations(raw: str) -> list[str]:
    """
    Parses the model's JSON array, tolerating markdown fences around it.
    Returns an empty list if nothing usable comes back.
    """
    if not raw:
        return []

    text = raw.strip()

    # Strip ```json ... ``` fences the model adds despite being told not to
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Unparseable expansion response | raw={raw[:120]!r}")
        return []

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, str) and item.strip()]
