from functools import lru_cache

import numpy as np

from app.config import CHROMA_PATH, EMBEDDING_MODEL
from app.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedder():
    """
    Loads the sentence-transformer model on first use.

    Kept out of module scope because loading it downloads (and pins ~90MB
    of) model weights — importing this module must stay free of network
    access and heavy I/O.
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model | model={EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_client():
    """Opens the persistent ChromaDB client on first use."""
    import chromadb
    from chromadb.config import Settings

    logger.info(f"Opening ChromaDB | path={CHROMA_PATH}")
    return chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
    )


def embed(texts: str | list[str]) -> np.ndarray:
    """Encodes a string or list of strings into embedding vectors."""
    return get_embedder().encode(texts)


def get_collection(tenant_id: str):
    """Each tenant has its own collection."""
    return get_chroma_client().get_or_create_collection(
        name=f"knowledge_{tenant_id}", metadata={"hnsw:space": "cosine"}
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
