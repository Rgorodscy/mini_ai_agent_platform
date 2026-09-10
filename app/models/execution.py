import uuid
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, func

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

    # --- Usage accounting ---
    #
    # Totals across every LLM call the run made, including the ones the RAG
    # pipeline makes internally. Nullable because rows written before this
    # existed have no usage data, and because a provider need not report
    # token counts at all.
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    llm_calls = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    # Null when any model involved has no configured price — never 0.0,
    # which would read as "this run was free".
    cost_usd = Column(Float, nullable=True)
    usage_by_model = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), index=True)
    updated_at = Column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
