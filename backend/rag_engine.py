import faiss
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("embeddings/faiss_index.bin")
with open("embeddings/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def rewrite_query(query: str):

    prompt = f"""
You are a restaurant search optimizer.

Convert the user's request into a retrieval-friendly sentence.

Rules:
- Preserve meaning
- Expand vague phrases
- Include cuisine, mood, dining style if implied
- Keep under 20 words
- Return ONLY the rewritten sentence

Examples:

User: cheap dosa place
Output: Affordable South Indian restaurants serving dosa

User: date night place with good vibes
Output: Romantic restaurants with cozy ambience suitable for date nights

User query:
{query}
"""

    try:
        response = ollama.chat(
            model="phi",
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )

        rewritten = response["message"]["content"].strip()

        return rewritten

    except:
        return query


def retrieve(query:str, top_k=5):
    optimized_query = rewrite_query(query)

    print(f"\nOriginal: {query}")
    print(f"Optimized: {optimized_query}\n")

    q_emb = model.encode(
        [optimized_query]
    ).astype("float32")

    distances, indices = index.search(
        q_emb,
        top_k*3
    )

    seen = set()
    results=[]

    for i in indices[0]:

        if i>=len(metadata):
            continue

        key=(
            metadata[i]["name"],
            metadata[i]["location"]
        )

        if key not in seen:
            seen.add(key)
            results.append(metadata[i])

        if len(results)==top_k:
            break

    return results

def build_prompt(query: str, results: list) -> str:
    context = "\n".join(
        f"{i+1}. {r['name']} ({r['location']}) | {r['cuisines']} | Rating: {r['rate']}"
        for i, r in enumerate(results)
    )

    return f"""
You are FoodGPT.

STRICT RULES:
- Use ONLY the restaurants below
- Do NOT add extra text
- Do NOT explain locations
- Keep output SHORT

User query: "{query}"

Restaurants:
{context}

Return EXACTLY this format (no extra lines before/after):

1. Name (Location) - short reason
2. Name (Location) - short reason
3. Name (Location) - short reason
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