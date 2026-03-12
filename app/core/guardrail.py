import re
from fastapi import HTTPException, status
from app.logger import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions",
    r"disregard\s+(previous|prior|above|all)\s+instructions",
    r"forget\s+(previous|prior|above|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are|a)\s+",
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


def check_prompt_injection(text: str) -> None:
    """
    Scans input text for prompt injection patterns.
    Raises HTTP 400 if a potential injection is detected.
    """
    for pattern in COMPILED_PATTERNS:
        if pattern.search(text):
            logger.warning(
                f"Prompt injection detected | pattern={pattern.pattern!r} "
                f"input_preview={text[:80]!r}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input rejected: potential prompt injection detected"
            )
