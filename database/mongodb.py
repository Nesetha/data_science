import os
from dotenv import load_dotenv
load_dotenv()

def get_database():
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI is not configured. Local JSON storage is used by default.")
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[os.getenv("MONGODB_DATABASE", "prenatal_ai")]
