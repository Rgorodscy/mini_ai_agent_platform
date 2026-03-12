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
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PaginatedExecutionResponse(BaseModel):
    total: int
    page: int
    size: int
    executions: list[ExecutionResponse]
