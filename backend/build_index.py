import pandas as pd
import faiss
import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

df = pd.read_csv("data/zomato.csv", encoding="latin-1")
df = df.dropna(subset=["name", "cuisines", "rate", "location"])

# Build a rich text chunk per restaurant
def build_chunk(row):
    return (
        f"{row['name']} in {row['location']}. "
        f"Cuisines: {row['cuisines']}. "
        f"Rating: {row['rate']}. "
        f"Cost for two: ₹{row.get('approx_cost(for two people)', 'N/A')}. "
        f"Type: {row.get('rest_type', '')}."
    )

df["chunk"] = df.apply(build_chunk, axis=1)
chunks = df["chunk"].tolist()

# ── Dense embeddings (FAISS) ──
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "embeddings/faiss_index.bin")

# ── Sparse embeddings (BM25) ──
tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# Save BM25 index data (corpus + parameters)
with open("embeddings/bm25_corpus.pkl", "wb") as f:
    pickle.dump({
        "tokenized_corpus": tokenized_chunks,
        "corpus": chunks,
    }, f)

# ── Metadata ──
metadata_records = df[["name", "location", "cuisines", "rate", "chunk"]].to_dict("records")
with open("embeddings/metadata.pkl", "wb") as f:
    pickle.dump(metadata_records, f)

print(f"Indexed {len(chunks)} restaurants (dense FAISS + sparse BM25).")