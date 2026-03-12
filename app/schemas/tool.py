from pydantic import BaseModel, ConfigDict, field_validator
from datetime import datetime
from typing import Optional


class ToolCreate(BaseModel):
    name: str
    description: str

    @field_validator("name", "description")
    @classmethod
    def must_not_be_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        if len(v) > 200:
            raise ValueError(f"{info.field_name} cannot exceed 200 characters")
        return v.strip()


class ToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

    @field_validator("name", "description")
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


class ToolResponse(BaseModel):
    id: str
    tenant_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
