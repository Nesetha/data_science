
from rag.retrieval import retrieve, answer_question

questions = [
    "What is spinal dysraphism and how does it develop?",
    "What role does prenatal ultrasound play in detecting spinal dysraphism?",
    "What are the advanced antenatal imaging techniques used for spinal dysraphism?"
]

for question in questions:

    print("\n" + "=" * 80)
    print("QUESTION:", question)

    docs, mode = retrieve(question)

    print("\nRETRIEVAL MODE:", mode)
    print("RETRIEVED CHUNKS:", len(docs))

    for doc in docs:
        print("\nSOURCE:", doc.get("citation", doc.get("source")))
        print("TEXT:", doc.get("text", "")[:350])

    result = answer_question(question)

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print("-", source)

    print("\nANSWER MODE:", result["mode"])
