from sentence_transformers import CrossEncoder
from state import GraphState

# Load cross-encoder model once at module level
# ms-marco-MiniLM-L-6-v2 is lightweight (~80MB) and fast for reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_node(state: GraphState) -> GraphState:
    """
    Rerank the hybrid-retrieved documents using a cross-encoder.
    Cross-encoder scores (query, doc) pairs for much more accurate relevance.
    """
    query = state["rewritten_query"]
    docs = state["retrieved_docs"]

    if not docs or len(docs) == 0:
        return {
            "reranked_docs": [],
            "results": [],
        }

    # Prepare (query, doc_text) pairs
    pairs = [(query, doc["chunk"]) for doc in docs]

    # Get relevance scores from cross-encoder
    scores = cross_encoder.predict(pairs)

    # Sort docs by score descending
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Take top 5 after reranking
    reranked = [doc for doc, score in scored_docs[:5]]

    print(f"Reranked: {len(docs)} candidates → {len(reranked)} final (scores: {[round(float(s), 3) for _, s in scored_docs[:5]]})")

    return {
        "reranked_docs": reranked,
        "results": reranked,
    }