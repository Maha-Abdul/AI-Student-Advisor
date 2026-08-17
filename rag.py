import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from ingest import load_documents, create_index


INDEX_FILE = Path("faiss_index.bin")
METADATA_FILE = Path("metadata.pkl")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def ensure_vector_store():
    """
    Build the FAISS index automatically if it does not exist.
    """
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():

        print("Vector store not found. Building index...")

        documents = load_documents()

        if not documents:
            raise RuntimeError(
                "No documents were found in the data folder."
            )

        create_index(documents)


def load_vector_store():

    ensure_vector_store()

    index = faiss.read_index(str(INDEX_FILE))

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
