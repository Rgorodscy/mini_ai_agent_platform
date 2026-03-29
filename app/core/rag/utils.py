import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


embedder = SentenceTransformer(EMBEDDING_MODEL)

_chroma_client = chromadb.PersistentClient(
    path="./chroma_db", settings=Settings(anonymized_telemetry=False)
)


def get_collection(tenant_id: str):
    """Each tenant has it's own collection"""
    return _chroma_client.get_or_create_collection(
        name=f"knowledge_{tenant_id}", metadata={"hnsw:space": "cosine"}
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
