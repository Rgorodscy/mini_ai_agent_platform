from pydantic import BaseModel, field_validator


class IngestRequest(BaseModel):
    doc_id: str
    text: str
    metadata: dict = {}

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text cannot be empty")
        if len(v) > 50000:
            raise ValueError("text cannot exceed 50000 characters")
        return v.strip()

    @field_validator("doc_id")
    @classmethod
    def doc_id_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("doc_id cannot be empty")
        return v.strip()


class IngestResponse(BaseModel):
    doc_id: str
    status: str
    message: str
