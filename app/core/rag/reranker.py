from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, context: list[str], top_k: int = 3) -> list[str]:
    pairs = [(query, c) for c in context]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(context, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in reranked[:top_k]]
