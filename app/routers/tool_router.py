from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.core.tool_implementations import TOOL_REGISTRY
from app.database import get_db
from app.middleware.auth import get_tenant
from app.schemas.models import AvailableTool
from app.services.tool_service import ToolService
from app.schemas.tool import ToolCreate, ToolUpdate, ToolResponse

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post("", response_model=ToolResponse, status_code=201)
def create_tool(
    data: ToolCreate,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    return ToolService(db).create_tool(tenant_id=tenant_id, data=data)


# Registered before /{tool_id}: routes match in order, and "available" would
# otherwise be taken as a tool id and answered with a 404.
@router.get("/available", response_model=list[AvailableTool])
def list_available_tools(tenant_id: str = Depends(get_tenant)):
    """
    The tool names that have an implementation on this platform.

    A tool row only becomes callable if its name is one of these; any other
    name is accepted at creation and silently skipped at run time. Clients
    should offer this list rather than free text.
    """
    return [
        AvailableTool(name=name, description=spec.get("description", ""))
        for name, spec in sorted(TOOL_REGISTRY.items())
    ]


@router.get("/{tool_id}", response_model=ToolResponse)
def get_tool(
    tool_id: str,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    return ToolService(db).get_tool(tool_id=tool_id, tenant_id=tenant_id)


@router.get("", response_model=list[ToolResponse])
def get_all_tools(
    tenant_id: str = Depends(get_tenant),
    agent_name: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return ToolService(db).get_all_tools(
        tenant_id=tenant_id, agent_name=agent_name
    )


@router.put("/{tool_id}", response_model=ToolResponse)
def update_tool(
    tool_id: str,
    data: ToolUpdate,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    return ToolService(db).update_tool(
        tool_id=tool_id, tenant_id=tenant_id, data=data
    )


@router.delete("/{tool_id}", status_code=204)
def delete_tool(
    tool_id: str,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    ToolService(db).delete_tool(tool_id=tool_id, tenant_id=tenant_id)
