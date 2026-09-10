from app.core.rag.utils import embed, get_collection
from app.logger import get_logger

logger = get_logger(__name__)


def retrieve(query: str, tenant_id: str, n_results: int = 3) -> list[str]:
    """
    Searches the tenant's collection for the chunks closest to the query.
    Returns the chunk texts.
    """
    collection = get_collection(tenant_id)

    count = collection.count()
    if count == 0:
        logger.info(f"Empty collection | tenant={tenant_id}")
        return []

    query_embedding = embed([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding, n_results=min(n_results, count)
    )

    documents = results.get("documents", [[]])[0]
    logger.info(
        f"Retrieved | tenant={tenant_id} query='{query[:50]}' "
        f"results={len(documents)}"
    )
    return documents
