"""
Measures what query expansion and reranking actually contribute.

Both add latency and tokens: expansion costs an LLM call per query, and
reranking runs a cross-encoder over every candidate. This harness exists so
those costs can be justified with numbers from this corpus instead of the
assumption that more pipeline is better.

Four configurations, same corpus and queries:

    baseline    vector search only
    +expansion  LLM rewrites the query into variations, results merged
    +rerank     cross-encoder reorders the candidates
    +both       the pipeline as shipped

Metrics are document-level. A query counts as answered at k if the document
labelled relevant appears among the first k distinct documents returned.

    recall@k   fraction of queries whose relevant document is in the top k
    MRR        mean reciprocal rank of the relevant document (0 if absent)

Usage:
    python -m evals.retrieval_eval
    python -m evals.retrieval_eval --no-llm     # offline, skips expansion
    python -m evals.retrieval_eval --k 1 3 5

--no-llm substitutes a deterministic stand-in for expansion so the harness
can be exercised without an API key. The expansion columns are then not
measuring a real LLM and are labelled accordingly.
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass

from app.core.rag.reranker import rerank
from app.core.rag.retriever import retrieve_with_metadata
from evals.dataset import CORPUS, QUERIES, Query

# Candidates pulled before reranking. Retrieval widens here so the
# cross-encoder has something to narrow; raising it trades latency for
# recall.
CANDIDATE_POOL = 10

# A tenant id of its own, so the harness never reads or writes a real
# tenant's collection. ChromaDB requires collection names to start and end
# with an alphanumeric character, which `get_collection` does not validate —
# so "eval-harness", not "__eval__".
EVAL_TENANT = "eval-harness"


@dataclass
class Result:
    name: str
    recall_at: dict[int, float]
    mrr: float
    seconds: float
    llm_calls: int
    note: str = ""


def ingest_corpus(tenant_id: str) -> None:
    from app.core.rag.indexer import ingest

    for doc_id, text in CORPUS.items():
        ingest(text, tenant_id=tenant_id, doc_id=doc_id)


def reset_corpus(tenant_id: str) -> None:
    """Drops and rebuilds the eval collection so runs are comparable."""
    from app.core.rag.utils import get_chroma_client

    client = get_chroma_client()
    try:
        client.delete_collection(f"knowledge_{tenant_id}")
    except Exception:
        pass
    ingest_corpus(tenant_id)


def _fake_expand(query: str) -> list[str]:
    """
    Offline stand-in for expansion: a few mechanical rewrites.

    Not a substitute for the real thing — it cannot add vocabulary the query
    does not contain, which is most of what expansion is for. It exists only
    so the harness runs without an API key.
    """
    words = [w for w in query.lower().strip("?").split() if len(w) > 3]
    variations = [query]
    if words:
        variations.append(" ".join(words))
    if len(words) > 2:
        variations.append(" ".join(words[:3]))
    return variations


def _distinct_docs(hits: list[dict]) -> list[str]:
    """Document ids in rank order, keeping only the best hit per document."""
    seen: dict[str, None] = {}
    for hit in hits:
        doc_id = hit.get("doc_id")
        if doc_id is not None:
            seen.setdefault(doc_id, None)
    return list(seen)


def _rank_of(relevant_doc: str, ranked_docs: list[str]) -> int | None:
    """1-based rank of the relevant document, or None if absent."""
    for position, doc_id in enumerate(ranked_docs, start=1):
        if doc_id == relevant_doc:
            return position
    return None


def _retrieve_for(
    query: Query,
    tenant_id: str,
    use_expansion: bool,
    use_rerank: bool,
    expand_fn,
) -> tuple[list[str], int]:
    """Returns (ranked doc ids, number of LLM calls made)."""
    llm_calls = 0

    if use_expansion:
        variations = expand_fn(query.text)
        llm_calls += 1
    else:
        variations = [query.text]

    # Merge across variations, keeping the best (smallest) distance per
    # chunk so the merged list stays ordered by relevance rather than by
    # which variation happened to run first.
    best: dict[str, dict] = {}
    for variation in variations:
        for hit in retrieve_with_metadata(
            variation, tenant_id, n_results=CANDIDATE_POOL
        ):
            existing = best.get(hit["text"])
            if existing is None or (hit["distance"] or 0) < (
                existing["distance"] or 0
            ):
                best[hit["text"]] = hit

    candidates = sorted(
        best.values(), key=lambda h: h["distance"] if h["distance"] else 0
    )

    if use_rerank and candidates:
        by_text = {hit["text"]: hit for hit in candidates}
        ordered_texts = rerank(
            query.text,
            [hit["text"] for hit in candidates],
            top_k=len(candidates),
        )
        candidates = [by_text[text] for text in ordered_texts]

    return _distinct_docs(candidates), llm_calls


def evaluate(
    name: str,
    tenant_id: str,
    ks: list[int],
    use_expansion: bool,
    use_rerank: bool,
    expand_fn,
    note: str = "",
) -> Result:
    hits_at: dict[int, int] = {k: 0 for k in ks}
    reciprocal_ranks: list[float] = []
    llm_calls = 0
    started = time.perf_counter()

    for query in QUERIES:
        ranked, calls = _retrieve_for(
            query, tenant_id, use_expansion, use_rerank, expand_fn
        )
        llm_calls += calls

        rank = _rank_of(query.relevant_doc, ranked)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1

    elapsed = time.perf_counter() - started
    total = len(QUERIES)

    return Result(
        name=name,
        recall_at={k: hits_at[k] / total for k in ks},
        mrr=statistics.fmean(reciprocal_ranks),
        seconds=elapsed,
        llm_calls=llm_calls,
        note=note,
    )


def print_table(results: list[Result], ks: list[int]) -> None:
    headers = (
        ["configuration"]
        + [f"recall@{k}" for k in ks]
        + ["MRR", "seconds", "LLM calls"]
    )
    rows = [
        [r.name]
        + [f"{r.recall_at[k]:.2f}" for k in ks]
        + [f"{r.mrr:.3f}", f"{r.seconds:.1f}", str(r.llm_calls)]
        for r in results
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    print()
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))

    notes = [r for r in results if r.note]
    if notes:
        print()
        for r in notes:
            print(f"  {r.name}: {r.note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure retrieval quality across pipeline stages."
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="cut-offs for recall@k (default: 1 3 5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="use a deterministic stand-in for query expansion",
    )
    args = parser.parse_args(argv)
    ks = sorted(set(args.k))

    if args.no_llm:
        expand_fn = _fake_expand
        note = "deterministic stand-in, not a real LLM expansion"
    else:
        from app.core.rag.query_expander import expand_query

        expand_fn = expand_query
        note = ""

    print(
        f"Corpus: {len(CORPUS)} documents, {len(QUERIES)} labelled queries, "
        f"candidate pool {CANDIDATE_POOL}"
    )
    print("Loading models and indexing...")
    reset_corpus(EVAL_TENANT)

    results = [
        evaluate("baseline", EVAL_TENANT, ks, False, False, expand_fn),
        evaluate("+expansion", EVAL_TENANT, ks, True, False, expand_fn, note),
        evaluate("+rerank", EVAL_TENANT, ks, False, True, expand_fn),
        evaluate("+both", EVAL_TENANT, ks, True, True, expand_fn, note),
    ]

    print_table(results, ks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
