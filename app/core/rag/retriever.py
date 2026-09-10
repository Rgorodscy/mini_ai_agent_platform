from app.core.rag.utils import embed, get_collection
from app.logger import get_logger

logger = get_logger(__name__)


def retrieve_with_metadata(
    query: str, tenant_id: str, n_results: int = 3
) -> list[dict]:
    """
    Searches the tenant's collection and returns the closest chunks with
    their metadata: {"text", "doc_id", "chunk", "distance"}.

    The metadata is what makes document-level evaluation possible (see
    evals/) and is the hook a citation feature would need.
    """
    collection = get_collection(tenant_id)

    count = collection.count()
    if count == 0:
        logger.info(f"Empty collection | tenant={tenant_id}")
        return []

    query_embedding = embed([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0] or [{}] * len(documents)
    distances = (results.get("distances") or [[]])[0] or [None] * len(
        documents
    )

    logger.info(
        f"Retrieved | tenant={tenant_id} query='{query[:50]}' "
        f"results={len(documents)}"
    )

    return [
        {
            "text": text,
            "doc_id": (metadata or {}).get("doc_id"),
            "chunk": (metadata or {}).get("chunk"),
            "distance": distance,
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def retrieve(query: str, tenant_id: str, n_results: int = 3) -> list[str]:
    """
    Searches the tenant's collection for the chunks closest to the query.
    Returns the chunk texts.
    """
    return [
        hit["text"]
        for hit in retrieve_with_metadata(query, tenant_id, n_results)
    ]
