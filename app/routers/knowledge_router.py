from fastapi import APIRouter, status, Depends

from app.schemas.knowledge import IngestRequest, IngestResponse
from app.middleware.auth import get_tenant
from app.core.rag import ingest
from app.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post(
    "", status_code=status.HTTP_201_CREATED, response_model=IngestResponse
)
def ingest_document(data: IngestRequest, tenant_id: str = Depends(get_tenant)):
    logger.info(f"Ingest request | tenant={tenant_id} doc={data.doc_id}")
    ingest(
        text=data.text,
        tenant_id=tenant_id,
        doc_id=data.doc_id,
        metadata=data.metadata,
    )
    return IngestResponse(
        doc_id=data.doc_id,
        status="ingested",
        message=f"Document '{data.doc_id}' ingested successfully.",
    )
