import uuid
from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.orm import relationship

from app.database import Base


class Tool(Base):
    __tablename__ = "tools"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now())

    agents = relationship("Agent",
                          secondary="agent_tools",
                          back_populates="tools")
