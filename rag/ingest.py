
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from src.database import get_database

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "research"

COLLECTION_NAME = "research_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_pdfs():

    pdf_files = list(RESEARCH_DIR.glob("*.pdf"))

    if not pdf_files:
        return {
            "success": False,
            "message": "No PDF files found in research/."
        }

    print("Loading embedding model...", flush=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    collection = get_database()[COLLECTION_NAME]
    total_chunks = 0

    for pdf_path in pdf_files:

        print(f"Processing: {pdf_path.name}", flush=True)

        reader = PdfReader(str(pdf_path))
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            chunks = splitter.split_text(text)

            for chunk_number, chunk in enumerate(chunks):

                documents.append({
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number
                })

        if not documents:
            print(f"No extractable text: {pdf_path.name}")
            continue

        vectors = embeddings.embed_documents(
            [doc["text"] for doc in documents]
        )

        records = []

        for doc, vector in zip(documents, vectors):

            records.append({
                "_id": str(uuid4()),
                "text": doc["text"],
                "source": doc["source"],
                "page": doc["page"],
                "chunk": doc["chunk"],
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL
            })

        # Replace previous chunks from the same PDF
        ingestion_id = str(uuid4())

        for record in records:
            record["ingestion_id"] = ingestion_id

        # Insert the new chunks first
        collection.insert_many(records)

        # Remove older chunks only after successful insertion
        collection.delete_many({
            "source": pdf_path.name,
            "ingestion_id": {"$ne": ingestion_id}
        })

        total_chunks += len(records)

        print(
            f"Stored {len(records)} chunks from {pdf_path.name}",
            flush=True
        )

    return {
        "success": True,
        "message": f"Successfully indexed {total_chunks} chunks.",
        "chunks": total_chunks,
        "documents": len(pdf_files)
    }


if __name__ == "__main__":

    result = ingest_pdfs()

    print("\nINGESTION RESULT")
    print(result)
