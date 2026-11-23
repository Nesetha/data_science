
from src.database import get_database

collection = get_database()["research_documents"]

docs = collection.find(
    {
        "source": "Antenatal_Spinal_Dysraphisms_2025.pdf",
        "page": 1
    },
    {
        "_id": 0,
        "text": 1,
        "chunk": 1,
        "source": 1,
        "page": 1
    }
)

for doc in docs:
    print("\n" + "=" * 70)
    print("CHUNK:", doc.get("chunk"))
    print("SOURCE:", doc.get("source"))
    print("PAGE:", doc.get("page"))
    print("TEXT:")
    print(doc.get("text", ""))
