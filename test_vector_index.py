
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid
from pymongo.operations import SearchIndexModel

from src.database import MONGODB_URI, DATABASE_NAME

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000
)

db = client[DATABASE_NAME]
collection_name = "research_documents"

# Create the collection if it does not exist
if collection_name not in db.list_collection_names():
    db.create_collection(collection_name)

collection = db[collection_name]

index_definition = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine"
        }
    ]
}

index_name = "prenatalai_vector_index"

existing_indexes = list(collection.list_search_indexes())

if any(index["name"] == index_name for index in existing_indexes):
    print("Vector index already exists.")
else:
    model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
        type="vectorSearch"
    )

    result = collection.create_search_index(model=model)
    print("Vector index creation requested:", result)

print("Collection:", collection_name)
print("Database:", DATABASE_NAME)

client.close()
