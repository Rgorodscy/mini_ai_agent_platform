from app.config import (
    GROQ_MODEL,
    RAG_CANDIDATE_POOL,
    RAG_TOP_K,
    RAG_USE_EXPANSION,
    RAG_USE_RERANK,
)
from app.core.llm import chat_completion
from app.core.rag.query_expander import expand_query
from app.core.rag.reranker import rerank
from app.core.rag.retriever import retrieve
from app.logger import get_logger

logger = get_logger(__name__)

RAG_PROMPT = """
You are a virtual assistant.
Answer concisely, based only on the text chunks below.
If the chunks do not contain what is needed to answer, say so plainly.
Never invent information that is not in the chunks.

Relevant chunks:
{context}
"""

NO_RESULTS = "No relevant document found in the knowledge base."


def answer_from_knowledge(query: str, tenant_id: str) -> str:
    """
    Answers a query from the tenant's knowledge base.

    Retrieval widens the candidate set; reranking narrows it. Both stages
    are optional (see config.RAG_USE_EXPANSION / RAG_USE_RERANK) because
    whether they earn their latency depends on the corpus — measure with
    evals/retrieval_eval.py rather than assuming.
    """
    if RAG_USE_EXPANSION:
        queries = expand_query(query)
    else:
        queries = [query]

    # dict.fromkeys semantics: preserve first-seen order while deduplicating
    # chunks that several query variations retrieved in common.
    seen: dict[str, None] = {}
    for expanded in queries:
        for chunk in retrieve(
            query=expanded, tenant_id=tenant_id, n_results=RAG_CANDIDATE_POOL
        ):
            seen.setdefault(chunk, None)

    candidates = list(seen)
    if not candidates:
        logger.info(f"RAG no candidates | tenant={tenant_id}")
        return NO_RESULTS

    if RAG_USE_RERANK:
        top_chunks = rerank(query, candidates, top_k=RAG_TOP_K)
    else:
        top_chunks = candidates[:RAG_TOP_K]

    logger.info(
        f"RAG context built | tenant={tenant_id} "
        f"expansion={RAG_USE_EXPANSION} rerank={RAG_USE_RERANK} "
        f"variations={len(queries)} candidates={len(candidates)} "
        f"kept={len(top_chunks)}"
    )

    context = "\n\n".join(top_chunks)

    res = chat_completion(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": RAG_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
    )

    return res.choices[0].message.content
