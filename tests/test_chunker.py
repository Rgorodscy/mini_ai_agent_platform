import pytest

from app.core.rag.chunker import (
    chunk_by_paragraph,
    recursive_chunk,
    semantic_chunk,
)
from tests.conftest import FakeEmbedder

embed = FakeEmbedder().encode


# --- chunk_by_paragraph ---


def test_paragraph_chunking_splits_on_blank_lines():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."

    assert chunk_by_paragraph(text) == [
        "First paragraph.",
        "Second paragraph.",
        "Third.",
    ]


def test_paragraph_chunking_drops_empty_chunks():
    assert chunk_by_paragraph("One.\n\n\n\n   \n\nTwo.") == ["One.", "Two."]


# --- recursive_chunk ---


def test_short_text_is_one_chunk():
    assert recursive_chunk("Short text.", max_length=500) == ["Short text."]


def test_every_chunk_respects_max_length():
    text = " ".join(f"Sentence number {i}." for i in range(200))

    chunks = recursive_chunk(text, max_length=200)

    assert all(len(c) <= 200 for c in chunks)


def test_text_without_punctuation_is_hard_split():
    """No paragraph or sentence boundary to split on — falls to characters."""
    text = "x" * 1000

    chunks = recursive_chunk(text, max_length=100)

    assert len(chunks) == 10
    assert all(len(c) == 100 for c in chunks)


def test_overlap_prepends_tail_of_previous_chunk():
    text = "abcdefghij" * 10

    chunks = recursive_chunk(text, max_length=50, overlap=5)

    # Every chunk after the first starts with the last 5 chars of the one
    # before it, so a fact split across a boundary stays retrievable.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.startswith(previous[-5:])


def test_no_content_is_lost():
    text = " ".join(f"Sentence {i}." for i in range(50))

    chunks = recursive_chunk(text, max_length=100)

    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


# --- semantic_chunk ---


def test_semantic_chunk_returns_the_whole_text_when_it_fits():
    text = "One sentence only."

    assert semantic_chunk(text, embed, max_length=800) == [text]


def test_semantic_chunk_respects_max_length():
    text = " ".join(f"This is sentence number {i}." for i in range(100))

    chunks = semantic_chunk(text, embed, max_length=200)

    assert all(len(c) <= 200 for c in chunks)


def test_semantic_chunk_groups_multiple_sentences():
    text = "Alpha beta. Alpha beta gamma. Alpha beta delta."

    chunks = semantic_chunk(text, embed, max_length=800)

    assert len(chunks) < 3


def test_semantic_chunk_ignores_empty_input():
    assert semantic_chunk("", embed) == []


def test_oversized_single_sentence_terminates():
    """
    Regression: an oversized chunk with no sentence boundary used to
    recurse into semantic_chunk forever, because splitting it produced
    itself. Extracted PDF text (tables, slides, bullet lists) often has no
    sentence punctuation, so this was reachable from /knowledge.
    """
    text = "word " * 400  # 2000 chars, not a single period

    chunks = semantic_chunk(text, embed, max_length=800)

    assert all(len(c) <= 800 for c in chunks)
    assert len(chunks) >= 3


@pytest.mark.parametrize("length", [801, 1600, 5000])
def test_oversized_sentences_of_various_lengths_terminate(length):
    chunks = semantic_chunk("x" * length, embed, max_length=800)

    assert all(len(c) <= 800 for c in chunks)
