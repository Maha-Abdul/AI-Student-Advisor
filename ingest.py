from pathlib import Path
import pickle
import requests
from bs4 import BeautifulSoup

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "metadata.pkl"

WEBSITE_URLS = [
    "https://nati.edu/",
    "https://nati.edu/admission-requirements/",
    "https://nati.edu/tuition-fees/",
    "https://nati.edu/registration/",
    "https://nati.edu/students-advisement/",
    "https://nati.edu/graduation/",
    "https://nati.edu/accreditation-and-approvals/",
    "https://nati.edu/contact-us/",
    "https://nati.edu/campus/",
    "https://nati.edu/our-mission-and-vision/",
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "text": text,
                "source": file_path.name,
                "page": page_number,
                "url": None,
            })

    return pages


def read_website(url):
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return {
        "text": text,
        "source": "NATI Website",
        "page": None,
        "url": url,
    }


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_documents():
    documents = []

    # Load catalog PDFs
    for pdf_file in DATA_DIR.glob("*.pdf"):
        pages = read_pdf(pdf_file)

        for page in pages:
            for chunk in chunk_text(page["text"]):
                documents.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "url": page["url"],
                })

    # Load NATI website pages
    for url in WEBSITE_URLS:
        try:
            page = read_website(url)

            for chunk in chunk_text(page["text"]):
                documents.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": None,
                    "url": url,
                })

            print(f"Loaded website: {url}")

        except Exception as e:
            print(f"Could not load {url}: {e}")

    return documents


def create_index(documents):
    texts = [doc["text"] for doc in documents]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
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
        print("No documents found.")
    else:
        create_index(documents)
