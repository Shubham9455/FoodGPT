import faiss
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss_index.bin")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def retrieve(query: str, top_k: int = 5):
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [metadata[i] for i in indices[0]]

def build_prompt(query: str, results: list) -> str:
    context = "\n\n".join(
        f"- {r['name']} ({r['location']}): {r['cuisines']}, "
        f"Rating {r['rate']}"
        for r in results
    )
    return (
        f"You are FoodGPT, a helpful restaurant recommendation assistant.\n"
        f"A user is asking: \"{query}\"\n\n"
        f"Here are some relevant restaurants from the Zomato database:\n{context}\n\n"
        f"Based on these options, give a helpful, conversational recommendation. "
        f"Mention specific restaurants by name and explain why they suit the user's request."
    )

def ask(query: str) -> str:
    results = retrieve(query)
    prompt = build_prompt(query, results)
    response = ollama.chat(
        model="phi",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"], results