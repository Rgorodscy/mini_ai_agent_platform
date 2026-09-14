from pydantic import BaseModel


class ModelsResponse(BaseModel):
    default: str
    models: list[str]


class AvailableTool(BaseModel):
    name: str
    description: str
