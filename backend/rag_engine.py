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
    distances, indices = index.search(q_emb, top_k * 3)

    seen = set()
    results = []

    for i in indices[0]:
        if i < len(metadata):
            key = (metadata[i]["name"], metadata[i]["location"])

            if key not in seen:
                seen.add(key)
                results.append(metadata[i])

        if len(results) == top_k:
            break

    return results

def build_prompt(query: str, results: list) -> str:
    context = "\n\n".join(
        f"- {r['name']} ({r['location']}): {r['cuisines']}, Rating {r['rate']}"
        for r in results
    )

    return f"""
You are FoodGPT, an AI restaurant recommendation assistant.

You MUST ONLY use the restaurants provided below.
Do NOT say you don't have access to data.
Do NOT mention limitations.

User query: "{query}"

Restaurants:
{context}

Task:
- Recommend 2-3 best options
- Explain WHY they match (cuisine, price, rating)
- Keep response concise and confident
"""

def ask(query: str):
    results = retrieve(query)

    if not results:
        return "No relevant restaurants found.", []

    prompt = build_prompt(query, results)

    try:
        response = ollama.chat(
            model="phi",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"], results

    except Exception as e:
        return "Something went wrong while generating response.", results