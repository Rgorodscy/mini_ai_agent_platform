from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional


from app.database import get_db
from app.middleware.auth import get_tenant
from app.services.agent_service import AgentService
from app.schemas.agent import AgentCreate, AgentUpdate, AgentResponse

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentResponse, status_code=201)
def create_agent(
    data: AgentCreate,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    return AgentService(db).create_agent(tenant_id=tenant_id, agent_data=data)


@router.get("/{agent_id}", response_model=AgentResponse)
def get_agent(
    agent_id: str,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    return AgentService(db).get_agent(tenant_id=tenant_id, agent_id=agent_id)


@router.get("", response_model=list[AgentResponse])
def get_agents(
    tenant_id: str = Depends(get_tenant),
    tool_name: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return AgentService(db).get_agents(tenant_id=tenant_id,
                                       tool_name=tool_name)


@router.put("/{agent_id}", response_model=AgentResponse)
def update_agent(
    agent_id: str,
    data: AgentUpdate,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    return AgentService(db).update_agent(
                                        agent_id=agent_id,
                                        tenant_id=tenant_id,
                                        data=data)


@router.delete("/{agent_id}", status_code=204)
def delete_agent(
    agent_id: str,
    tenant_id: str = Depends(get_tenant),
    db: Session = Depends(get_db),
):
    AgentService(db).delete_agent(agent_id=agent_id, tenant_id=tenant_id)
