import json
from typing import Optional

from fastapi import (
    APIRouter,
    status,
    Depends,
    Form,
    UploadFile,
    File,
    HTTPException,
)

from app.schemas.knowledge import IngestResponse
from app.middleware.auth import get_tenant
from app.core.rag import ingest
from app.core.utils import extract_text_from_file
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post(
    "", status_code=status.HTTP_201_CREATED, response_model=IngestResponse
)
async def ingest_document(
    doc_id: str = Form(...),
    text: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    tenant_id: str = Depends(get_tenant),
):
    if not text and not file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide either 'text' or a file upload.",
        )

    if text and file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Provide either 'text' or a file — not both.",
        )

    parsed_metadata: dict = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="'metadata' must be a valid JSON string.",
            )

    final_text = text if text else await extract_text_from_file(file)

    logger.info(
        f"Ingest request | tenant={tenant_id} doc={doc_id} "
        f"source={'file' if file else 'text'}"
    )

    ingest(
        text=final_text,
        tenant_id=tenant_id,
        doc_id=doc_id,
        metadata=parsed_metadata,
    )

    return IngestResponse(
        doc_id=doc_id,
        status="ingested",
        message=f"Document '{doc_id}' ingested successfully.",
    )


"""
---

**Pontos-chave para você entender:**

**Por que `Form()` e não Pydantic model?** Porque `multipart/form-data` (necessário para upload) e `application/json` são content-types mutuamente exclusivos no HTTP. O FastAPI não consegue ler um Pydantic body JSON quando o request é multipart.

**Por que `metadata` como string JSON?** Em `multipart/form-data`, todos os campos são strings. Então você serializa o dict no cliente (`JSON.stringify`) e desserializa no servidor com `json.loads`.

**Por que `async def`?** O `await file.read()` exige isso — sem ele o FastAPI travaria esperando I/O de arquivo.

**Dependências a adicionar:**
pip install pypdf python-docx
"""
