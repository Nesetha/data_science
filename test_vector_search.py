
from rag.vector_store import search_similar

questions = [
    "What is spinal dysraphism and how does it develop?",
    "What role does prenatal ultrasound play in detecting spinal dysraphism?",
    "What are the advanced antenatal imaging techniques used for spinal dysraphism?"
]

for question in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", question)
    print("=" * 70)

    results = search_similar(question, k=4)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source:", doc.get("source"))
        print("Page:", doc.get("page"))
        print("Score:", round(doc.get("score", 0), 4))
        print("Text:", doc.get("text", "")[:450])
