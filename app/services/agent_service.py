from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import Optional

from app.repositories.agent_repo import AgentRepository
from app.repositories.tool_repo import ToolRepository
from app.schemas.agent import AgentCreate, AgentUpdate, AgentResponse
from app.logger import get_logger

logger = get_logger(__name__)


class AgentService:
    def __init__(self, db: Session):
        self.agent_repo = AgentRepository(db)
        self.tool_repo = ToolRepository(db)

    def _resolve_tools(self, tool_ids: list[str], tenant_id: str) -> list:
        tools = []
        for tool_id in tool_ids:
            tool = self.tool_repo.get_by_id(tool_id, tenant_id)
            if not tool:
                logger.warning(
                    f"Tool not found during agent operation | "
                    f"tool_id={tool_id} tenant={tenant_id}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Tool {tool_id} not found"
                )
            tools.append(tool)
        return tools

    def create_agent(
            self,
            tenant_id: str,
            agent_data: AgentCreate) -> AgentResponse:
        tools = self._resolve_tools(agent_data.tools, tenant_id)
        agent = self.agent_repo.create(
            tenant_id=tenant_id,
            name=agent_data.name,
            role=agent_data.role,
            description=agent_data.description,
            tools=tools
        )
        logger.info(f"Agent created | agent_id={agent.id} tenant={tenant_id}")
        return AgentResponse.model_validate(agent)

    def get_agent(self, tenant_id: str, agent_id: str) -> AgentResponse:
        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(
                f"Agent not found | agent_id={agent_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        return AgentResponse.model_validate(agent)

    def get_agents(
            self,
            tenant_id: str,
            tool_name: Optional[str] = None) -> list[AgentResponse]:
        agents = self.agent_repo.get_all(tenant_id, tool_name)
        return [AgentResponse.model_validate(agent) for agent in agents]

    def update_agent(
            self,
            agent_id: str,
            tenant_id: str,
            data: AgentUpdate) -> AgentResponse:
        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(
                f"Agent not found | agent_id={agent_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        tools = (self._resolve_tools(data.tools, tenant_id) if
                 data.tools is not None else None)
        updated = self.agent_repo.update(
            agent,
            data.name,
            data.role,
            data.description,
            tools)
        logger.info(f"Agent updated | agent_id={agent_id} tenant={tenant_id}")
        return AgentResponse.model_validate(updated)

    def delete_agent(self, agent_id: str, tenant_id: str) -> None:
        agent = self.agent_repo.get_by_id(agent_id, tenant_id)
        if not agent:
            logger.warning(
                f"Agent not found | agent_id={agent_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found"
            )
        logger.info(f"Agent deleted | agent_id={agent_id} tenant={tenant_id}")
        self.agent_repo.delete(agent)
