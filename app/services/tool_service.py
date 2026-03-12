from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import Optional

from app.repositories.tool_repo import ToolRepository
from app.schemas.tool import ToolCreate, ToolUpdate, ToolResponse
from app.logger import get_logger

logger = get_logger(__name__)


class ToolService:
    def __init__(self, db: Session):
        self.repo = ToolRepository(db)

    def create_tool(self, tenant_id: str, data: ToolCreate) -> ToolResponse:
        tool = self.repo.create(
            tenant_id=tenant_id,
            name=data.name,
            description=data.description
        )
        logger.info(f"Tool created | tool_id={tool.id} tenant={tenant_id}")
        return ToolResponse.model_validate(tool)

    def get_tool(self, tool_id: str, tenant_id: str) -> ToolResponse:
        tool = self.repo.get_by_id(tool_id=tool_id, tenant_id=tenant_id)

        if not tool:
            logger.warning(
                f"Tool not found | tool_id={tool_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {tool_id} not found")

        return ToolResponse.model_validate(tool)

    def get_all_tools(self, tenant_id: str, agent_name: Optional[str] = None
                      ) -> list[ToolResponse]:
        tools = self.repo.get_all(tenant_id=tenant_id, agent_name=agent_name)
        return [ToolResponse.model_validate(tool) for tool in tools]

    def update_tool(self,
                    tool_id: str,
                    tenant_id: str,
                    data: ToolUpdate) -> ToolResponse:
        tool = self.repo.get_by_id(tool_id=tool_id, tenant_id=tenant_id)
        if not tool:
            logger.warning(
                f"Tool not found | tool_id={tool_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {tool_id} not found")
        updated_tool = self.repo.update(
            tool=tool,
            name=data.name,
            description=data.description
        )
        logger.info(f"Tool updated | tool_id={tool_id} tenant={tenant_id}")
        return ToolResponse.model_validate(updated_tool)

    def delete_tool(self, tool_id: str, tenant_id: str) -> None:
        tool = self.repo.get_by_id(tool_id=tool_id, tenant_id=tenant_id)
        if not tool:
            logger.warning(
                f"Tool not found | tool_id={tool_id} tenant={tenant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {tool_id} not found")
        logger.info(f"Tool deleted | tool_id={tool_id} tenant={tenant_id}")
        self.repo.delete(tool)
