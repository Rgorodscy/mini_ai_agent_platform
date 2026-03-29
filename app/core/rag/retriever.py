from app.logger import get_logger
from app.core.rag.utils import get_collection, embedder

logger = get_logger(__name__)


def retrieve(query: str, tenant_id: str, n_results: int = 3) -> list[str]:
    """
    Searches most relevant chunks to the query
    Returns list of text
    """
    collection = get_collection(tenant_id)

    count = collection.count()
    if count == 0:
        logger.info(f"Empty collection | tenant={tenant_id}")
        return []

    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding, n_results=min(n_results, count)
    )

    documents = results.get("documents", [])[0]
    logger.info(
        f"Retrieved | tenant={tenant_id} query='{query[:50]}' results={len(documents)}"
    )
    return documents
