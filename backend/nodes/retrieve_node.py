import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.state import GraphState

# Load shared resources once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss_index.bin")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant restaurant documents from FAISS index."""
    query = state["rewritten_query"]
    top_k = 5

    q_emb = model.encode([query]).astype("float32")

    distances, indices = index.search(q_emb, top_k * 3)

    seen = set()
    results = []

    for i in indices[0]:
        if i >= len(metadata):
            continue

        key = (metadata[i]["name"], metadata[i]["location"])

        if key not in seen:
            seen.add(key)
            results.append(metadata[i])

        if len(results) == top_k:
            break

    return {
        "retrieved_docs": results,
        "results": results,
    }