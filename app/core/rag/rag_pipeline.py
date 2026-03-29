from app.core.rag.retriever import retrieve
from app.core.rag.reranker import rerank
from app.core.rag.query_expander import expand_query
from app.config import groq_client


RAG_PROMPT = """
You are a virtual assistant.
Respond in a concise way, based only on the text chunks below.
If you don't find relevant chunks of text to help your analysis,
acknoledge that and respond accordingly, don't create data\n\n
Relevant chunks:\n{context}",
"""


def rag_pipeline(query: str, tenant_id: str) -> str:
    expanded_queries = expand_query(query)
    relevant_messages = []
    for e_q in expanded_queries:
        messages = retrieve(query=e_q, tenant_id=tenant_id, n_results=10)
        relevant_messages.extend(messages)
    seen = set()
    deduplicated = []
    for msg in relevant_messages:
        if msg not in seen:
            seen.add(msg)
            deduplicated.append(msg)
    relevant_messages = deduplicated
    if not relevant_messages:
        return "No document found based on the filters used."
    relevant_messages = rerank(query, relevant_messages)
    context = "".join(relevant_messages)
    res = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": RAG_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
    )

    return res.choices[0].message.content
