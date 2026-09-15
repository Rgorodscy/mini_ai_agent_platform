"""
Indexing and retrieval against a real ChromaDB instance.

The embedding model is faked (see conftest) but ChromaDB itself is real
and runs against a throwaway directory per test, so collection naming,
upsert semantics and tenant scoping are exercised for real rather than
mocked away.
"""

from app.core.rag.indexer import ingest
from app.core.rag.retriever import retrieve, retrieve_with_variations
from app.core.rag.utils import get_collection
from tests.conftest import fake_search, make_hit

DOC = (
    "The refund policy allows returns within 30 days.\n\n"
    "Shipping is free above fifty euros.\n\n"
    "Support is available on weekdays."
)


# --- Ingestion ---


def test_ingest_stores_chunks():
    ingest(DOC, tenant_id="t1", doc_id="policy")

    assert get_collection("t1").count() > 0


def test_ingest_records_doc_id_and_chunk_index():
    ingest(DOC, tenant_id="t1", doc_id="policy")

    stored = get_collection("t1").get()

    assert all(m["doc_id"] == "policy" for m in stored["metadatas"])
    assert sorted(m["chunk"] for m in stored["metadatas"]) == list(
        range(len(stored["ids"]))
    )


def test_ingest_merges_caller_metadata():
    ingest(DOC, tenant_id="t1", doc_id="policy", metadata={"source": "wiki"})

    stored = get_collection("t1").get()

    assert all(m["source"] == "wiki" for m in stored["metadatas"])


def test_ingest_handles_missing_metadata():
    ingest(DOC, tenant_id="t1", doc_id="policy", metadata=None)

    assert get_collection("t1").count() > 0


def test_reingesting_same_doc_id_replaces_chunks():
    ingest(DOC, tenant_id="t1", doc_id="policy")
    first_count = get_collection("t1").count()

    ingest(DOC, tenant_id="t1", doc_id="policy")

    assert get_collection("t1").count() == first_count


def test_ingesting_a_second_doc_adds_to_the_collection():
    ingest(DOC, tenant_id="t1", doc_id="policy")
    first_count = get_collection("t1").count()

    ingest("A completely different document.", tenant_id="t1", doc_id="other")

    assert get_collection("t1").count() > first_count


def test_ingest_of_unsplittable_text_still_stores_something():
    ingest("x" * 3000, tenant_id="t1", doc_id="blob")

    assert get_collection("t1").count() > 0


# --- Retrieval ---


def test_retrieve_on_empty_collection_returns_empty():
    assert retrieve("anything", tenant_id="empty-tenant") == []


def test_retrieve_returns_chunk_texts():
    ingest(DOC, tenant_id="t1", doc_id="policy")

    results = retrieve("refund policy", tenant_id="t1", n_results=3)

    assert results
    assert all(isinstance(r, str) for r in results)


def test_retrieve_respects_n_results():
    ingest(DOC, tenant_id="t1", doc_id="policy")

    assert len(retrieve("refund", tenant_id="t1", n_results=1)) == 1


def test_retrieve_caps_n_results_at_collection_size():
    """Asking for more than exists must not raise."""
    ingest("Just one short chunk.", tenant_id="t1", doc_id="tiny")

    results = retrieve("chunk", tenant_id="t1", n_results=100)

    assert len(results) == get_collection("t1").count()


# --- Tenant isolation ---


def test_tenants_get_separate_collections():
    assert get_collection("tenant_a").name != get_collection("tenant_b").name


def test_one_tenant_cannot_retrieve_another_tenants_document():
    ingest(
        "Tenant A's confidential salary table.",
        tenant_id="tenant_a",
        doc_id="secret",
    )

    assert retrieve("salary table", tenant_id="tenant_b") == []


def test_ingesting_for_one_tenant_leaves_the_other_empty():
    ingest(DOC, tenant_id="tenant_a", doc_id="policy")

    assert get_collection("tenant_b").count() == 0


def test_same_doc_id_in_two_tenants_does_not_collide():
    ingest("Tenant A version.", tenant_id="tenant_a", doc_id="shared-id")
    ingest("Tenant B version.", tenant_id="tenant_b", doc_id="shared-id")

    a = retrieve("version", tenant_id="tenant_a")
    b = retrieve("version", tenant_id="tenant_b")

    assert a == ["Tenant A version."]
    assert b == ["Tenant B version."]


def test_vector_store_is_isolated_between_tests():
    """
    Guards the fixture that gives each test its own Chroma directory: if
    this leaked, the assertions above would pass or fail depending on test
    order.
    """
    assert get_collection("t1").count() == 0


# --- Merging hits across query variations ---


def _merge(*variations):
    return retrieve_with_variations(
        list(variations), tenant_id="t1", n_results=10
    )


def test_variations_merge_by_smallest_distance(monkeypatch):
    fake_search(
        monkeypatch,
        {
            "original": [make_hit("A", 0.50), make_hit("B", 0.60)],
            "variation": [make_hit("C", 0.10), make_hit("A", 0.30)],
        },
    )

    hits = _merge("original", "variation")

    assert [h["text"] for h in hits] == ["C", "A", "B"]
    assert [h["distance"] for h in hits] == [0.10, 0.30, 0.60]
    assert [h["doc_id"] for h in hits] == ["doc-C", "doc-A", "doc-B"]


def test_missing_distance_ranks_last_and_perfect_match_first(monkeypatch):
    """
    A perfect match has distance 0.0, which is falsy: ranking by
    `distance or inf` would send it last alongside the hit with no distance.
    """
    fake_search(
        monkeypatch,
        {
            "original": [
                make_hit("A", None),
                make_hit("B", 0.0),
                make_hit("C", 0.40),
            ]
        },
    )

    hits = _merge("original")

    assert [h["text"] for h in hits] == ["B", "C", "A"]
    assert [h["distance"] for h in hits] == [0.0, 0.40, None]
    assert [h["doc_id"] for h in hits] == ["doc-B", "doc-C", "doc-A"]


def test_a_real_distance_replaces_a_missing_one(monkeypatch):
    fake_search(
        monkeypatch,
        {
            "original": [make_hit("A", None), make_hit("B", 0.20)],
            "variation": [make_hit("A", 0.10)],
        },
    )

    hits = _merge("original", "variation")

    assert [(h["text"], h["distance"]) for h in hits] == [
        ("A", 0.10),
        ("B", 0.20),
    ]


def test_a_missing_distance_does_not_replace_a_real_one(monkeypatch):
    """
    Regression guard: comparing the incoming None against a stored distance
    raises TypeError, so a missing distance must never enter that branch.
    """
    fake_search(
        monkeypatch,
        {
            "original": [make_hit("A", 0.0), make_hit("B", 0.20)],
            "variation": [make_hit("A", None)],
        },
    )

    hits = _merge("original", "variation")

    assert [(h["text"], h["distance"]) for h in hits] == [
        ("A", 0.0),
        ("B", 0.20),
    ]


def test_merging_does_not_modify_the_hits_it_receives(monkeypatch):
    first_a = make_hit("A", 0.50)
    fake_search(
        monkeypatch,
        {"original": [first_a], "variation": [make_hit("A", 0.30)]},
    )

    _merge("original", "variation")

    assert first_a["distance"] == 0.50
