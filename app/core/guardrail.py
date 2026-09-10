import re

from fastapi import HTTPException, status

from app.logger import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions",
    r"disregard\s+(previous|prior|above|all)\s+instructions",
    r"forget\s+(previous|prior|above|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    # Narrow: "act as a reviewer" is a legitimate task, so this only fires
    # on the jailbreak phrasings that follow "act as".
    r"act\s+as\s+(if\s+you\s+(are\s+not|were\s+not|have\s+no)|an?\s+"
    r"(unrestricted|unfiltered|jailbroken|uncensored))",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"override\s+(your\s+)?(instructions|rules|guidelines)",
    r"system\s*prompt\s*:",
    r"<\s*system\s*>",
    r"import\s+os",
    r"exec\s*\(",
    r"eval\s*\(",
    r"__import__\s*\(",
    r"subprocess",
    r"system\s*\(",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> str | None:
    """
    Scans text for prompt and code injection patterns.
    Returns the pattern that matched, or None if the text looks clean.

    This is the detection primitive: callers decide what to do about a
    match. HTTP callers want a 400 (see check_prompt_injection); the
    execution loop wants to block a single tool call and let the agent
    carry on.
    """
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def check_prompt_injection(text: str) -> None:
    """
    Guards an inbound request. Raises HTTP 400 if an injection is detected.
    """
    matched = detect_injection(text)
    if matched:
        logger.warning(
            f"Prompt injection detected | pattern={matched!r} "
            f"input_preview={text[:80]!r}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input rejected: potential prompt injection detected",
        )
