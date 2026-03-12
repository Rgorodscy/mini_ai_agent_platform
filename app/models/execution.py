import uuid
from sqlalchemy import Column, String, DateTime, JSON, func

from app.database import Base


class Execution(Base):
    __tablename__ = "executions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String, nullable=False, index=True)
    agent_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    task = Column(String, nullable=False)
    structured_prompt = Column(JSON, nullable=False)
    steps = Column(JSON, nullable=False, default=list)
    final_response = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now())
