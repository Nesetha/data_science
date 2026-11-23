
from pathlib import Path
import re
import json

from rag.knowledge_base import BUNDLED_KNOWLEDGE
from src.safety import validate_question, check_response, validate_citations
ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data" / "research_index.json"


# --------------------------------------------------
# LOCAL RETRIEVAL — FALLBACK
# --------------------------------------------------

def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _load_index():
    if INDEX_FILE.exists():
        try:
            return json.loads(
                INDEX_FILE.read_text(encoding="utf-8")
            )
        except Exception:
            return []

    return []


def _local_retrieve(question, k=4):
    """
    Keyword-based retrieval used only when
    MongoDB Vector Search is unavailable or empty.
    """

    docs = list(BUNDLED_KNOWLEDGE) + _load_index()

    query_words = _tokens(question)
    ranked = []

    for doc in docs:
        text = doc.get("text", "")
        words = _tokens(text)

        score = len(query_words & words) / max(len(query_words), 1)

        if score > 0:
            ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)

    return [doc for _, doc in ranked[:k]]


# --------------------------------------------------
# VECTOR RETRIEVAL — MONGODB ATLAS
# --------------------------------------------------

def _vector_retrieve(question, k=4):
    """
    Retrieves semantically relevant research chunks
    from MongoDB Atlas Vector Search.
    """

    from rag.vector_store import search_similar

    results = search_similar(question, k=k)

    normalized = []

    for item in results:
        doc = dict(item)

        # Preserve source and page metadata.
        source = doc.get("source", "Unknown source")
        page = doc.get("page")

        if page is not None:
            doc["citation"] = f"{source}, page {page}"
        else:
            doc["citation"] = str(source)

        normalized.append(doc)

    return normalized


# --------------------------------------------------
# MAIN RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve(question, k=4):
    """
    MongoDB Vector Search is the primary retriever.
    Local keyword retrieval is the fallback.
    """

    try:
        vector_docs = _vector_retrieve(question, k)

        if vector_docs:
            return vector_docs, "mongodb vector search"

    except Exception as exc:
        print(f"[RAG] Vector retrieval unavailable: {exc}")

    local_docs = _local_retrieve(question, k)

    if local_docs:
        return local_docs, "local keyword fallback"

    return [], "no retrieval"


# --------------------------------------------------
# SOURCE FORMATTING
# --------------------------------------------------

def _format_sources(docs):
    """
    Creates a unique list of source references.
    """

    sources = []

    for doc in docs:
        citation = doc.get("citation")

        if not citation:
            citation = doc.get("source", "Unknown source")

            if doc.get("page") is not None:
                citation = f"{citation}, page {doc['page']}"

        if citation not in sources:
            sources.append(citation)

    return sources


def _build_context(docs):
    """
    Adds source labels to retrieved passages so
    the language model can cite its evidence.
    """

    passages = []

    for index, doc in enumerate(docs, start=1):
        citation = doc.get("citation") or doc.get(
            "source", "Unknown source"
        )

        text = doc.get("text", "").strip()

        if text:
            passages.append(
                f"[S{index}] SOURCE: {citation}\n{text}"
            )

    return "\n\n".join(passages)


# --------------------------------------------------
# OLLAMA ANSWER GENERATION
# --------------------------------------------------

def _generate_with_ollama(question, context):
    """
    Generates an answer using only the retrieved
    research passages.
    """

    from dotenv import load_dotenv
    import os
    import requests

    load_dotenv(ROOT / ".env")

    model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    prompt = f"""
You are PrenatalAI, an evidence-grounded medical research assistant.

Your task is to answer questions using ONLY the research passages
provided below.

STRICT EVIDENCE RULES:

1. Every factual claim must be directly supported by the passages.
2. Never use pretrained knowledge or outside medical information.
3. Never invent imaging techniques, procedures, statistics,
   clinical recommendations, or research findings.
4. Do not infer that a technique is established merely because
   the passage mentions it as an emerging possibility.
5. If a technique is mentioned without sufficient explanation,
   state exactly what the passage says and identify the limitation.
6. If the passages do not contain the requested information,
   explicitly say: "The available research passages do not
   provide sufficient information to answer this question."
7. Never fill missing information using assumptions.
8. Every factual statement must end with at least one
   supporting citation, such as [S1].
9. Use ONLY the source labels actually provided in the passages.
10. Never cite a source unless it directly supports the claim.
11. If a claim is supported by multiple passages, cite them
    together, for example [S1][S2].
12. Do not invent citations or page numbers.
13. If no passage supports a claim, omit that claim.

ANSWER FORMAT:

- Avoid repeating the same finding in different words.
- Use a maximum of 3 numbered points unless more are necessary.
- Every point must introduce a distinct finding.
- Do not make stronger claims than the source passages support.
- Do not describe a method as first-line or definitive unless
  the retrieved passages explicitly establish that status.
- Keep evidence limitations separate from established findings.

MEDICAL SAFETY:
This is an educational research assistant, not a diagnostic
system. Do not provide patient-specific diagnoses or treatment
recommendations.

RESEARCH PASSAGES:
{context}

USER QUESTION:
{question}

EVIDENCE-BASED ANSWER:
"""

    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=90
    )

    response.raise_for_status()

    return response.json().get("response", "").strip()


# --------------------------------------------------
# MAIN QUESTION ANSWERING FUNCTION
# --------------------------------------------------

def answer_question(question):

    # 1. Validate the question.
    valid, reason = validate_question(question)

    if not valid:
        return {
            "answer": reason,
            "sources": [],
            "mode": "validation"
        }

    # 2. Retrieve relevant research passages.
    docs, retrieval_mode = retrieve(question)

    if not docs:
        return {
            "answer": (
                "I could not find relevant material in the "
                "available research library. Please add "
                "appropriate research sources or consult "
                "qualified clinical literature."
            ),
            "sources": [],
            "mode": retrieval_mode
        }

    # 3. Prepare context and source references.
    context = _build_context(docs)
    sources = _format_sources(docs)

    # 4. Try optional Ollama generation.
    try:
        from dotenv import load_dotenv
        import os

        load_dotenv(ROOT / ".env")

        provider = os.getenv(
            "LLM_PROVIDER", "ollama"
        ).lower()

        if provider == "ollama":
            generated = _generate_with_ollama(
                question,
                context
            )

           
            if generated:
                # Check for unsafe diagnostic wording.
                safety_result = check_response(generated)

                if not safety_result["approved"]:
                    return {
                        "answer": (
                            "The generated response was blocked "
                            "because potentially unsafe diagnostic "
                            "wording was detected. Please consult "
                            "a qualified healthcare professional."
                        ),
                        "sources": sources,
                        "mode": "safety blocked"
                    }

                # Check that all citation labels exist.
                citation_result = validate_citations(
                    generated, context
                )

                if not citation_result["approved"]:
                    return {
                        "answer": (
                            "The generated response contained "
                            "invalid source references and could "
                            "not be displayed reliably."
                        ),
                        "sources": sources,
                        "mode": "citation validation blocked"
                    }

                return {
                    "answer": generated,
                    "sources": sources,
                    "mode": f"ollama + {retrieval_mode}"
                }


    except Exception as exc:
        print(f"[RAG] Answer generation unavailable: {exc}")

    # 5. Safe fallback when Ollama is unavailable.
    answer = (
        "Based on the retrieved research passages:\n\n"
        + context
        + "\n\nEvidence limitation: This response summarizes "
        "retrieved research material. It is not a clinical "
        "diagnosis or a substitute for professional medical "
        "assessment."
    )

    safety_result = check_response(answer)

    if not safety_result["approved"]:
        answer = (
            "The fallback response was blocked by the safety checker. "
            "Please consult a qualified healthcare professional."
        )

    return {
        "answer": answer,
        "sources": sources,
        "mode": retrieval_mode
    }


# --------------------------------------------------
# LEGACY PDF PAGE INDEXING
# --------------------------------------------------

def index_research_folder():
    """
    Retained for compatibility with the existing
    local keyword-retrieval fallback.
    """

    try:
        from pypdf import PdfReader

        folder = ROOT / "research"
        docs = []

        pdf_files = list(folder.glob("*.pdf"))

        for path in pdf_files:
            reader = PdfReader(str(path))

            for page_num, page in enumerate(
                reader.pages, start=1
            ):
                text = page.extract_text() or ""

                if text.strip():
                    docs.append({
                        "text": text[:12000],
                        "source": f"{path.name}, page {page_num}"
                    })

        INDEX_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        INDEX_FILE.write_text(
            json.dumps(docs, indent=2),
            encoding="utf-8"
        )

        return (
            len(docs),
            f"Indexed {len(docs)} PDF pages "
            f"from {len(pdf_files)} files."
        )

    except Exception as exc:
        return -1, f"Indexing failed: {exc}"
