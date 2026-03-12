from sqlalchemy.orm import Session
from typing import Optional

from app.models.tool import Tool
from app.models.agent import Agent


class ToolRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, tenant_id: str, name: str, description: str) -> Tool:
        tool = Tool(tenant_id=tenant_id, name=name, description=description)
        self.db.add(tool)
        self.db.commit()
        self.db.refresh(tool)
        return tool

    def get_by_id(self, tool_id: str, tenant_id: str) -> Optional[Tool]:
        return (
            self.db.query(Tool)
            .filter(Tool.id == tool_id, Tool.tenant_id == tenant_id)
            .first()
        )

    def get_all(
            self,
            tenant_id: str,
            agent_name: Optional[str] = None) -> list[Tool]:
        query = self.db.query(Tool).filter(Tool.tenant_id == tenant_id)

        if agent_name:
            query = (
                query
                .join(Tool.agents)
                .filter(Agent.name.ilike(f"%{agent_name}%")))

        return query.all()

    def update(
            self,
            tool: Tool,
            name: Optional[str],
            description: Optional[str]) -> Tool:
        if name is not None:
            tool.name = name
        if description is not None:
            tool.description = description

        self.db.commit()
        self.db.refresh(tool)
        return tool

    def delete(self, tool: Tool) -> None:
        self.db.delete(tool)
        self.db.commit()
