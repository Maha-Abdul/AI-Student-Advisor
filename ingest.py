from pathlib import Path
import pickle

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "text": text,
                    "source": file_path.name,
                    "page": page_number,
                }
            )

    return pages


def read_txt(file_path):
    text = file_path.read_text(encoding="utf-8")

    return [
        {
            "text": text,
            "source": file_path.name,
            "page": None,
        }
    ]


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_documents():
    documents = []

    for file_path in DATA_DIR.iterdir():

        if file_path.suffix.lower() == ".pdf":
            pages = read_pdf(file_path)

        elif file_path.suffix.lower() == ".txt":
            pages = read_txt(file_path)

        else:
            continue

        for page in pages:
            chunks = chunk_text(page["text"])

            for chunk in chunks:
                documents.append(
                    {
                        "text": chunk,
                        "source": page["source"],
                        "page": page["page"],
                    }
                )

    return documents


def create_index(documents):
    texts = [doc["text"] for doc in documents]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "wb") as f:
        pickle.dump(documents, f)

    print(f"Indexed {len(documents)} document chunks.")


if __name__ == "__main__":
    documents = load_documents()

    if not documents:
        print("No PDF or TXT files found in the data folder.")
    else:
        create_index(documents)
