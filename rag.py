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
    Build the vector store automatically if it does not exist.
    """
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("Vector store not found. Building from catalog and website...")

        documents = load_documents()

        if not documents:
            raise RuntimeError(
                "No catalog or website documents were found."
            )

        create_index(documents)


def load_vector_store():
    ensure_vector_store()

    index = faiss.read_index(str(INDEX_FILE))

    with open(METADATA_FILE, "rb") as file:
        metadata = pickle.load(file)

    return index, metadata


def retrieve(question, top_k=5):
    """
    Retrieve the most relevant catalog and website chunks.
    """
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

        document = metadata[idx]

        results.append(
            {
                "text": document["text"],
                "source": document.get("source"),
                "page": document.get("page"),
                "url": document.get("url"),
                "score": float(score),
            }
        )

    return results


def build_context(results):
    """
    Combine retrieved information into context for the LLM.
    """
    context_parts = []

    for result in results:

        source_info = result["source"]

        if result.get("page"):
            source_info += f", page {result['page']}"

        if result.get("url"):
            source_info += f"\nURL: {result['url']}"

        context_parts.append(
            f"""
SOURCE: {source_info}

CONTENT:
{result['text']}
"""
        )

    return "\n\n".join(context_parts)


if __name__ == "__main__":

    test_question = "What are the admission requirements?"

    results = retrieve(test_question)

    print("\nQUESTION:")
    print(test_question)

    print("\nRETRIEVED RESULTS:")

    for result in results:
        print("\n----------------")
        print("Score:", result["score"])
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("URL:", result["url"])
        print(result["text"][:500])
