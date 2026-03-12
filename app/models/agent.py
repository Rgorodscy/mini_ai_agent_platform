import uuid
from sqlalchemy import Column, String, DateTime, Table, ForeignKey, func
from sqlalchemy.orm import relationship

from app.database import Base

agent_tools = Table(
    "agent_tools",
    Base.metadata,
    Column("agent_id", String, ForeignKey("agents.id"), primary_key=True),
    Column("tool_id", String, ForeignKey("tools.id"), primary_key=True),
)


class Agent(Base):
    __tablename__ = "agents"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)
    description = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now())

    tools = relationship(
        "Tool",
        secondary=agent_tools,
        back_populates="agents")
