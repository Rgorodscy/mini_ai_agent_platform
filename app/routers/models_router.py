from fastapi import APIRouter, Depends

from app.config import GROQ_MODEL
from app.core.llm import supported_models
from app.middleware.auth import get_tenant
from app.schemas.models import ModelsResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsResponse)
def list_models(tenant_id: str = Depends(get_tenant)):
    """
    The models a run may request on this deployment.

    Derived from the providers that have credentials configured, so a model
    listed here is one the API will accept — clients can offer this list
    instead of hardcoding names that may have been retired.
    """
    return ModelsResponse(default=GROQ_MODEL, models=supported_models())
