
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.middleware.auth import get_tenant
from app.services.execution_service import ExecutionService
from app.schemas.execution import (
    RunRequest,
    ExecutionResponse,
    PaginatedExecutionResponse
)

router = APIRouter(prefix="/agents", tags=["Executions"])


@router.post("/{agent_id}/run",
             response_model=ExecutionResponse,
             status_code=201)
def run_agent(
    agent_id: str,
    data: RunRequest,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db)
):
    return ExecutionService(db).run_agent(agent_id, tenant_id, data)


@router.get("/{agent_id}/history", response_model=PaginatedExecutionResponse)
def get_history(
    agent_id: str,
    page: int = Query(default=1,
                      ge=1,
                      description="Page number, starting from 1"),
    page_size: int = Query(default=10,
                           ge=1,
                           le=100,
                           description="Results per page, max 100"),
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db)
):
    return ExecutionService(db).get_history(
        agent_id, tenant_id, page, page_size)
