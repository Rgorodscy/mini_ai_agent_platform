import re
import numpy as np
from typing import List, Callable

from app.core.rag.utils import cosine_similarity


def chunk_by_paragraph(text: str) -> list[str]:
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks


def recursive_chunk(text, max_length=500, overlap=0):
    """
    Recursively splits text into chunks no longer than max_length.
    Splits first by paragraphs, then sentences, then characters if needed.
    """
    text = text.strip()

    # Base case: fits in one chunk
    if len(text) <= max_length:
        return [text]

    # Try splitting by paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        return _split_and_recurse(paragraphs, max_length, overlap)

    # Try splitting by sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        return _split_and_recurse(sentences, max_length, overlap)

    # Fallback: hard split by characters
    return _hard_split(text, max_length, overlap)


def _split_and_recurse(parts, max_length, overlap):
    """
    Takes a list of text parts (paragraphs or sentences) and groups them
    into chunks, then recursively ensures each chunk fits the size limit.
    """
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) + 1 <= max_length:
            current += " " + part if current else part
        else:
            chunks.extend(recursive_chunk(current, max_length, overlap))
            current = part

    if current:
        chunks.extend(recursive_chunk(current, max_length, overlap))

    # Add overlap if needed
    if overlap > 0:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def _hard_split(text, max_length, overlap):
    """
    Final fallback: split text by fixed character length.
    """
    chunks = [
        text[i : i + max_length] for i in range(0, len(text), max_length)
    ]

    if overlap > 0:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def _add_overlap(chunks, overlap):
    """
    Adds character-level overlap between consecutive chunks.
    """
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev = chunks[i - 1]
            prefix = prev[-overlap:]
            overlapped.append(prefix + chunk)
    return overlapped


def semantic_chunk(
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    max_length: int = 800,
    similarity_threshold: float = 0.65,
) -> List[str]:
    """
    Semantic chunking with recursive fallback.
    - embed_fn: function that returns a vector for a given string.
    - max_length: max characters per chunk.
    - similarity_threshold: minimum similarity to keep adding sentences.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current_chunk = []
    current_embedding = None

    for sentence in sentences:
        if not sentence:
            continue

        # If starting a new chunk
        if not current_chunk:
            current_chunk.append(sentence)
            current_embedding = embed_fn(sentence)
            continue

        # Check semantic similarity
        sentence_embedding = embed_fn(sentence)
        sim = cosine_similarity(current_embedding, sentence_embedding)

        # Check size and similarity constraints
        tentative = " ".join(current_chunk + [sentence])
        if len(tentative) > max_length or sim < similarity_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_embedding = sentence_embedding
        else:
            current_chunk.append(sentence)
            current_embedding = (current_embedding + sentence_embedding) / 2

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Recursively split oversized chunks
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_length:
            final_chunks.extend(
                semantic_chunk(
                    chunk, embed_fn, max_length, similarity_threshold
                )
            )
        else:
            final_chunks.append(chunk)

    return final_chunks
