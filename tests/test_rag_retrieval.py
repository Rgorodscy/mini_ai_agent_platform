"""
Indexing and retrieval against a real ChromaDB instance.

The embedding model is faked (see conftest) but ChromaDB itself is real
and runs against a throwaway directory per test, so collection naming,
upsert semantics and tenant scoping are exercised for real rather than
mocked away.
"""

from app.core.rag.indexer import ingest
from app.core.rag.retriever import retrieve
from app.core.rag.utils import get_collection

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
