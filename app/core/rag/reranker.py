from functools import lru_cache

from app.config import RERANKER_MODEL
from app.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_reranker():
    """Loads the cross-encoder on first use. See rag.utils.get_embedder."""
    from sentence_transformers import CrossEncoder

    logger.info(f"Loading reranker model | model={RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, context: list[str], top_k: int = 3) -> list[str]:
    """
    Scores each chunk against the query with a cross-encoder and returns
    the top_k most relevant ones, most relevant first.

    Unlike the embedding similarity used for retrieval, the cross-encoder
    sees the query and the chunk together, so it is far more accurate —
    and far too slow to run over the whole collection, which is why it
    only reranks what retrieval already narrowed down.
    """
    if not context:
        return []

    pairs = [(query, c) for c in context]
    scores = get_reranker().predict(pairs)

    reranked = sorted(zip(context, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:top_k]]
