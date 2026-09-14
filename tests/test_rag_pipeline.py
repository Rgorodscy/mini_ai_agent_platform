from app.core.rag.indexer import ingest
from app.core.rag.pipeline import NO_RESULTS, answer_from_knowledge
from app.core.rag.query_expander import expand_query
from app.core.rag.reranker import rerank
from tests.conftest import make_llm_response


def json_response(payload: str):
    """An LLM reply whose content is raw text (used for the JSON array)."""
    return make_llm_response(content=payload)


# --- Query expansion ---


def test_expansion_parses_a_json_array(fake_llm):
    fake_llm.queue(json_response('["refund policy", "returns", "money back"]'))

    assert expand_query("refund policy") == [
        "refund policy",
        "returns",
        "money back",
    ]


def test_expansion_always_includes_the_original_query(fake_llm):
    fake_llm.queue(json_response('["returns", "money back"]'))

    assert expand_query("refund policy")[0] == "refund policy"


def test_expansion_strips_markdown_fences(fake_llm):
    """The model adds ```json fences despite being told not to."""
    fake_llm.queue(json_response('```json\n["a", "b"]\n```'))

    assert expand_query("a") == ["a", "b"]


def test_expansion_falls_back_on_unparseable_output(fake_llm):
    fake_llm.queue(json_response("Sure! Here are some variations: a, b, c"))

    assert expand_query("original") == ["original"]


def test_expansion_falls_back_on_wrong_json_shape(fake_llm):
    fake_llm.queue(json_response('{"queries": ["a", "b"]}'))

    assert expand_query("original") == ["original"]


def test_expansion_drops_non_string_items(fake_llm):
    fake_llm.queue(json_response('["valid", 42, null, "  ", "also valid"]'))

    assert expand_query("q") == ["q", "valid", "also valid"]


def test_expansion_falls_back_when_the_llm_raises(fake_llm, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(fake_llm, "create", boom)

    assert expand_query("original") == ["original"]


def test_expansion_never_loses_the_query_on_empty_output(fake_llm):
    fake_llm.queue(json_response(""))

    assert expand_query("original") == ["original"]


# --- Reranking ---


def test_rerank_orders_by_relevance():
    chunks = [
        "completely unrelated content here",
        "the refund policy explained",
        "somewhat about refund",
    ]

    result = rerank("refund policy", chunks, top_k=3)

    assert result[0] == "the refund policy explained"


def test_rerank_limits_to_top_k():
    chunks = [f"chunk number {i}" for i in range(10)]

    assert len(rerank("chunk", chunks, top_k=3)) == 3


def test_rerank_handles_empty_context():
    assert rerank("anything", []) == []


def test_rerank_handles_fewer_chunks_than_top_k():
    assert len(rerank("q", ["only one chunk"], top_k=5)) == 1


# --- Full pipeline ---


def test_pipeline_returns_no_results_when_nothing_is_indexed(fake_llm):
    fake_llm.queue(json_response('["anything"]'))

    assert answer_from_knowledge("anything", "empty-tenant") == NO_RESULTS


def test_pipeline_does_not_call_the_llm_for_an_answer_when_empty(fake_llm):
    fake_llm.queue(json_response('["anything"]'))

    answer_from_knowledge("anything", "empty-tenant")

    # One call for expansion, none for generation.
    assert len(fake_llm.calls) == 1


def test_pipeline_grounds_the_answer_in_retrieved_chunks(fake_llm):
    ingest(
        "The refund policy allows returns within 30 days.",
        tenant_id="t1",
        doc_id="policy",
    )
    fake_llm.queue(
        json_response('["refund policy"]'),
        make_llm_response(content="Returns are accepted within 30 days."),
    )

    answer = answer_from_knowledge("refund policy", "t1")

    assert answer == "Returns are accepted within 30 days."

    # The generation call must carry the retrieved chunk as context.
    generation_call = fake_llm.calls[-1]
    system_prompt = generation_call["messages"][0]["content"]
    assert "30 days" in system_prompt


def test_pipeline_deduplicates_chunks_across_query_variations(fake_llm):
    """
    Every variation retrieves from the same small collection, so the same
    chunks come back repeatedly — they must reach the reranker once each.
    """
    ingest("Only one chunk of text here.", tenant_id="t1", doc_id="one")
    fake_llm.queue(
        json_response('["a", "b", "c", "d"]'),
        make_llm_response(content="Answer."),
    )

    answer_from_knowledge("query", "t1")

    system_prompt = fake_llm.calls[-1]["messages"][0]["content"]
    assert system_prompt.count("Only one chunk of text here.") == 1


def test_pipeline_is_tenant_scoped(fake_llm):
    ingest("Tenant A secret.", tenant_id="tenant_a", doc_id="secret")
    fake_llm.queue(json_response('["secret"]'))

    assert answer_from_knowledge("secret", "tenant_b") == NO_RESULTS


def test_pipeline_survives_a_failed_expansion(fake_llm):
    """A broken expansion degrades recall; it must not fail the search."""
    ingest("Refunds within 30 days.", tenant_id="t1", doc_id="policy")
    fake_llm.queue(
        json_response("not json at all"),
        make_llm_response(content="Within 30 days."),
    )

    assert answer_from_knowledge("refunds", "t1") == "Within 30 days."


# --- Stage toggles ---
#
# Whether expansion and reranking earn their latency depends on the corpus
# (see evals/retrieval_eval.py), so both are switchable. These verify the
# switches actually bypass the stage rather than just logging differently.


def test_expansion_can_be_disabled(fake_llm, monkeypatch):
    monkeypatch.setattr("app.core.rag.pipeline.RAG_USE_EXPANSION", False)
    ingest("Refunds within 30 days.", tenant_id="t1", doc_id="policy")
    fake_llm.queue(make_llm_response(content="Within 30 days."))

    answer_from_knowledge("refunds", "t1")

    # One call for the answer, none for expansion.
    assert len(fake_llm.calls) == 1


def test_rerank_can_be_disabled(monkeypatch, fake_llm):
    monkeypatch.setattr("app.core.rag.pipeline.RAG_USE_RERANK", False)

    called = []
    monkeypatch.setattr(
        "app.core.rag.pipeline.rerank",
        lambda *a, **kw: called.append(a) or [],
    )
    ingest("Refunds within 30 days.", tenant_id="t1", doc_id="policy")
    fake_llm.queue(
        json_response('["refunds"]'),
        make_llm_response(content="Within 30 days."),
    )

    answer_from_knowledge("refunds", "t1")

    assert called == []


def test_top_k_limits_context_without_rerank(monkeypatch, fake_llm):
    monkeypatch.setattr("app.core.rag.pipeline.RAG_USE_RERANK", False)
    monkeypatch.setattr("app.core.rag.pipeline.RAG_TOP_K", 1)

    for i in range(5):
        ingest(
            f"Unrelated document number {i}.", tenant_id="t1", doc_id=f"d{i}"
        )
    fake_llm.queue(
        json_response('["query"]'),
        make_llm_response(content="Answer."),
    )

    answer_from_knowledge("document", "t1")

    system_prompt = fake_llm.calls[-1]["messages"][0]["content"]
    assert system_prompt.count("Unrelated document number") == 1


def test_both_stages_disabled_still_answers(monkeypatch, fake_llm):
    monkeypatch.setattr("app.core.rag.pipeline.RAG_USE_EXPANSION", False)
    monkeypatch.setattr("app.core.rag.pipeline.RAG_USE_RERANK", False)
    ingest("Refunds within 30 days.", tenant_id="t1", doc_id="policy")
    fake_llm.queue(make_llm_response(content="Within 30 days."))

    assert answer_from_knowledge("refunds", "t1") == "Within 30 days."


# --- The tool the model actually calls ---


def test_search_knowledge_reports_no_results_without_contradiction(fake_llm):
    """
    Regression: with nothing indexed, the tool returned
    "Relevant knowledge for '...':\n\nNo relevant document found" — telling
    the model it had relevant knowledge and then that it had none. Found by
    a real run in which one tenant searched another tenant's documents.
    """
    from app.core.tool_implementations import make_search_knowledge

    fake_llm.queue(json_response('["refund window"]'))

    result = make_search_knowledge("empty-tenant")("refund window")

    assert "Relevant knowledge" not in result
    assert result == "No relevant documents found for: 'refund window'"


def test_search_knowledge_wraps_a_real_answer(fake_llm):
    from app.core.tool_implementations import make_search_knowledge

    ingest("Refunds within 30 days.", tenant_id="t1", doc_id="policy")
    fake_llm.queue(
        json_response('["refunds"]'),
        make_llm_response(content="Within 30 days."),
    )

    result = make_search_knowledge("t1")("refunds")

    assert result == "Relevant knowledge for 'refunds':\n\nWithin 30 days."
