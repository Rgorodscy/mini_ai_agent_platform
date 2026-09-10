from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.middleware.auth import get_tenant
from app.schemas.execution import UsageReportResponse
from app.services.execution_service import ExecutionService

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", response_model=UsageReportResponse)
def get_usage(
    since: Optional[datetime] = Query(
        default=None, description="Only count executions created at or after"
    ),
    until: Optional[datetime] = Query(
        default=None, description="Only count executions created at or before"
    ),
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    """
    LLM consumption for the calling tenant: tokens, call count, cost and
    average latency, broken down by model and by agent.

    The tenant comes from the API key, never from a parameter — there is no
    way to ask for another tenant's usage.
    """
    return ExecutionService(db).get_usage(tenant_id, since=since, until=until)
