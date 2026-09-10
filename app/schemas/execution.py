from pydantic import BaseModel, ConfigDict, field_validator
from datetime import datetime
from typing import Optional


class RunRequest(BaseModel):
    task: str
    model: str

    @field_validator("task")
    @classmethod
    def task_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("task cannot be empty")
        if len(v) > 2000:
            raise ValueError("task cannot exceed 2000 characters")
        return v.strip()


class ExecutionResponse(BaseModel):
    id: str
    tenant_id: str
    agent_id: str
    model: str
    task: str
    structured_prompt: dict
    steps: list
    final_response: Optional[str]
    status: str

    # Usage is null for rows written before accounting existed, and cost is
    # null whenever a model involved has no configured price.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    llm_calls: Optional[int] = None
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    usage_by_model: Optional[dict] = None

    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedExecutionResponse(BaseModel):
    total: int
    page: int
    size: int
    executions: list[ExecutionResponse]


class UsageTotals(BaseModel):
    executions: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls: int
    # Null, not zero, when no price is configured for a model involved.
    cost_usd: Optional[float] = None
    avg_latency_ms: Optional[int] = None


class ModelUsageBreakdown(UsageTotals):
    model: str

    # `model` collides with Pydantic's protected `model_` namespace, which
    # would otherwise emit a warning for this field name.
    model_config = ConfigDict(protected_namespaces=())


class AgentUsageBreakdown(UsageTotals):
    agent_id: str


class UsageReportResponse(UsageTotals):
    tenant_id: str
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    by_model: list[ModelUsageBreakdown] = []
    by_agent: list[AgentUsageBreakdown] = []
