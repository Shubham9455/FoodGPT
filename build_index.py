import pandas as pd
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

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

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True, batch_size=64)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "embeddings/faiss_index.bin")
with open("embeddings/metadata.pkl", "wb") as f:
    pickle.dump(df[["name", "location", "cuisines", "rate", "chunk"]].to_dict("records"), f)

print(f"Indexed {len(chunks)} restaurants.")