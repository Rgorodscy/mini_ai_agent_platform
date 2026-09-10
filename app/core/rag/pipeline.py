from app.config import GROQ_MODEL
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

    Retrieval is widened by query expansion, then narrowed by a
    cross-encoder rerank before the surviving chunks are handed to the
    model as grounding context.
    """
    expanded_queries = expand_query(query)

    # dict.fromkeys preserves first-seen order while deduplicating chunks
    # that several query variations retrieved in common.
    seen: dict[str, None] = {}
    for expanded in expanded_queries:
        for chunk in retrieve(
            query=expanded, tenant_id=tenant_id, n_results=10
        ):
            seen.setdefault(chunk, None)

    candidates = list(seen)
    if not candidates:
        logger.info(f"RAG no candidates | tenant={tenant_id}")
        return NO_RESULTS

    top_chunks = rerank(query, candidates)

    logger.info(
        f"RAG context built | tenant={tenant_id} "
        f"variations={len(expanded_queries)} candidates={len(candidates)} "
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
