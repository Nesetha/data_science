
import re
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from src.database import get_database

COLLECTION_NAME = "research_documents"
INDEX_NAME = "prenatalai_vector_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def clean_research_text(text):
    """Remove publication metadata and bibliography noise."""

    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []

    metadata_patterns = [
        r"^accepted:",
        r"^published online:",
        r"^©",
        r"^copyright",
        r"^all rights reserved",
        r"^https?://",
        r"^[\w.+-]+@[\w.-]+\.[a-z]{2,}$",
        r"^department of ",
        r"^university of ",
        r"^\d+\s+department of ",
        r"^\d+\s+university of "
    ]

    for line in lines:
        lower = line.lower()

        # Stop when a bibliography section begins.
        if lower in ("references", "bibliography", "key references"):
            break

        # Skip obvious publication metadata.
        if any(re.search(pattern, lower) for pattern in metadata_patterns):
            continue

        # Skip lines dominated by author contact information.
        if lower.count("@") >= 1:
            continue

        cleaned.append(line)

    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def is_low_quality_chunk(text):
    cleaned = clean_research_text(text)
    return len(cleaned) < 100


def remove_duplicate_chunks(results):
    unique = []
    seen = set()

    for doc in results:
        cleaned = clean_research_text(doc.get("text", ""))

        key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()

        if not key or key in seen:
            continue

        seen.add(key)

        item = dict(doc)
        item["text"] = cleaned
        unique.append(item)

    return unique


def search_similar(query, k=4):
    """Semantic retrieval with research-text cleanup."""

    if not query or not query.strip():
        return []

    model = get_embedding_model()
    query_vector = model.embed_query(query)

    collection = get_database()[COLLECTION_NAME]

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(k * 40, 200),
                "limit": max(k * 15, 60)
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "source": 1,
                "page": 1,
                "chunk": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    results = [
        doc for doc in results
        if not is_low_quality_chunk(doc.get("text", ""))
    ]

    results = remove_duplicate_chunks(results)

    # Preserve source diversity where relevant.
    selected = []
    seen_sources = set()

    for doc in results:
        source = doc.get("source")

        if source not in seen_sources:
            selected.append(doc)
            seen_sources.add(source)

        if len(selected) == k:
            return selected

    for doc in results:
        if doc not in selected:
            selected.append(doc)

        if len(selected) == k:
            break

    return selected
