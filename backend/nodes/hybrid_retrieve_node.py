import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from state import GraphState

# ── Load shared resources once at module level ──

# Dense (FAISS)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss_index.bin")

# Metadata
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# BM25
with open("embeddings/bm25_corpus.pkl", "rb") as f:
    bm25_data = pickle.load(f)
bm25 = BM25Okapi(bm25_data["tokenized_corpus"])


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    top_k: int = 10,
    k: int = 60,
) -> list:
    """
    Fuse dense and sparse results using Reciprocal Rank Fusion.
    RRF(d) = Σ 1/(k + r_i(d))  for each system i.
    """
    scores = {}

    for rank, doc in enumerate(dense_results, start=1):
        key = (doc["name"], doc["location"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank)

    for rank, doc in enumerate(sparse_results, start=1):
        key = (doc["name"], doc["location"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank)

    # Sort by fused score descending
    sorted_keys = sorted(scores, key=scores.get, reverse=True)

    # Map back to metadata dicts (deduplicated by name+location)
    seen = set()
    fused = []
    for key in sorted_keys:
        if key not in seen:
            seen.add(key)
            # Find the full metadata dict for this key
            for doc in dense_results:
                if (doc["name"], doc["location"]) == key:
                    fused.append(doc)
                    break
        if len(fused) >= top_k:
            break

    return fused


def hybrid_retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieve documents using:
      1. Dense retrieval (FAISS + sentence-transformers)
      2. Sparse retrieval (BM25)
      3. Fusion via Reciprocal Rank Fusion (RRF)
    """
    query = state["rewritten_query"]
    top_k = 10  # Retrieve more candidates for reranking later

    # ── 1. Dense retrieval ──
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k * 3)

    seen_dense = set()
    dense_results = []
    for i in indices[0]:
        if i >= len(metadata):
            continue
        key = (metadata[i]["name"], metadata[i]["location"])
        if key not in seen_dense:
            seen_dense.add(key)
            dense_results.append(metadata[i])
        if len(dense_results) == top_k:
            break

    # ── 2. Sparse retrieval (BM25) ──
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_sparse_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]

    seen_sparse = set()
    sparse_results = []
    for i in top_sparse_indices:
        if i >= len(metadata):
            continue
        key = (metadata[i]["name"], metadata[i]["location"])
        if key not in seen_sparse:
            seen_sparse.add(key)
            sparse_results.append(metadata[i])
        if len(sparse_results) == top_k:
            break

    # ── 3. Fuse with RRF ──
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)

    print(f"\nHybrid retrieval: {len(dense_results)} dense + {len(sparse_results)} sparse → {len(fused_results)} fused")

    return {
        "retrieved_docs": fused_results,
        "results": None,  # Will be set after reranking
    }