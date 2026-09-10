from app.core.rag.chunker import semantic_chunk
from app.core.rag.utils import embed, get_collection
from app.logger import get_logger

logger = get_logger(__name__)


def ingest(
    text: str, tenant_id: str, doc_id: str, metadata: dict = None
) -> None:
    """
    Chunks, embeds and stores a document in the tenant's collection.

    Re-ingesting the same doc_id upserts over the previous chunks.
    """
    chunks = semantic_chunk(text, embed)
    if not chunks:
        chunks = [text]

    collection = get_collection(tenant_id)

    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    embeddings = embed(chunks).tolist()
    metadatas = [
        {**(metadata or {}), "doc_id": doc_id, "chunk": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
    )

    logger.info(
        f"Ingested | tenant={tenant_id} doc={doc_id} chunks={len(chunks)}"
    )
