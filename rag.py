import pickle

import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_vector_store():
    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def retrieve(question, top_k=3):
    index, metadata = load_vector_store()

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        doc = metadata[idx]

        results.append(
            {
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
                "score": float(score),
            }
        )

    return results


if __name__ == "__main__":
    question = "Do you offer an artificial intelligence program?"

    results = retrieve(question)

    for result in results:
        print("\n---")
        print("Score:", result["score"])
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Text:", result["text"])
