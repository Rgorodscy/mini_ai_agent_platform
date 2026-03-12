from pydantic import BaseModel, ConfigDict, field_validator
from datetime import datetime
from typing import Optional

from app.schemas.tool import ToolResponse


class AgentCreate(BaseModel):
    name: str
    role: str
    description: str
    tools: list[str] = []

    @field_validator("name", "role", "description")
    @classmethod
    def must_not_be_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        if len(v) > 200:
            raise ValueError(f"{info.field_name} cannot exceed 200 characters")
        return v.strip()


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    description: Optional[str] = None
    tools: Optional[list[str]] = None

    @field_validator("name", "role", "description")
    @classmethod
    def must_not_be_empty_if_provided(cls, v: str, info) -> str:
        if v is not None:
            if not v.strip():
                raise ValueError(f"{info.field_name} cannot be empty")
            if len(v) > 200:
                raise ValueError(
                    f"{info.field_name} cannot exceed 200 characters")
            return v.strip()
        return v


class AgentResponse(BaseModel):
    id: str
    tenant_id: str
    name: str
    role: str
    description: str
    tools: list[ToolResponse] = []
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
