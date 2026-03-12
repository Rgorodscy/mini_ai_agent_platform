from fastapi import Header, HTTPException, status

from app.config import API_KEYS
from app.logger import get_logger

logger = get_logger(__name__)


def get_tenant(x_api_key: str = Header(..., description="Tenant API key")):
    """
    Extracts and validates the tenant API key from the request header.
    Returns the tenant_id associated with the key
    """

    tenant_id = API_KEYS.get(x_api_key)

    if not tenant_id:
        logger.warning(
            f"Invalid API key attempt | key_preview={x_api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    return tenant_id
