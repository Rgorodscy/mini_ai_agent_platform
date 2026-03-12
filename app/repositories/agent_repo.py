from sqlalchemy.orm import Session, joinedload
from typing import Optional

from app.models.agent import Agent
from app.models.tool import Tool


class AgentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(
            self,
            tenant_id: str,
            name: str,
            role: str,
            description: str,
            tools: list[Tool]) -> Agent:
        agent = Agent(
            tenant_id=tenant_id,
            name=name,
            role=role,
            description=description,
            tools=tools
        )
        self.db.add(agent)
        self.db.commit()
        self.db.refresh(agent)
        return agent

    def get_by_id(self, agent_id: str, tenant_id: str) -> Optional[Agent]:
        self.db.expire_all()
        return self.db.query(Agent).options(joinedload(Agent.tools)).filter(
            Agent.id == agent_id,
            Agent.tenant_id == tenant_id).first()

    def get_all(self, tenant_id: str, tool_name: Optional[str] = None
                ) -> list[Agent]:
        query = self.db.query(Agent).filter(Agent.tenant_id == tenant_id)

        if tool_name:
            query = query.join(Agent.tools).filter(
                Tool.name.ilike(f"%{tool_name}%")
            )

        return query.all()

    def update(
            self,
            agent: Agent,
            name: Optional[str] = None,
            role: Optional[str] = None,
            description: Optional[str] = None,
            tools: Optional[list[Tool]] = None) -> Agent:
        if name is not None:
            agent.name = name
        if role is not None:
            agent.role = role
        if description is not None:
            agent.description = description
        if tools is not None:
            agent.tools = tools

        self.db.commit()
        self.db.refresh(agent)
        return agent

    def delete(self, agent: Agent) -> None:
        self.db.delete(agent)
        self.db.commit()
