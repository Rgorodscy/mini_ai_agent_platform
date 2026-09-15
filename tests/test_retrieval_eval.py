"""
The retrieval eval's ranking.

An eval is only worth running if it measures the pipeline that ships. This
one once merged query variations its own way while production kept
first-seen order, and reported a gain production never had.
"""

from evals.dataset import Query
from evals.retrieval_eval import _retrieve_for
from tests.conftest import fake_search, make_hit


def test_eval_ranks_documents_with_the_production_merge(monkeypatch):
    fake_search(
        monkeypatch,
        {
            "original": [make_hit("A", 0.40), make_hit("B", 0.50)],
            "variation": [make_hit("C", 0.10)],
        },
    )

    ranked, llm_calls = _retrieve_for(
        Query("original", relevant_doc="doc-C"),
        tenant_id="t1",
        use_expansion=True,
        use_rerank=False,
        expand_fn=lambda text: [text, "variation"],
    )

    assert ranked == ["doc-C", "doc-A", "doc-B"]
    assert llm_calls == 1
