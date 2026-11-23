
import os
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "prenatalai_db")

_client = None


def get_database():
    global _client

    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI is missing. Check your project's .env file."
        )

    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000
        )

    return _client[DATABASE_NAME]


def test_connection():
    try:
        db = get_database()
        db.command("ping")

        return {
            "success": True,
            "database": DATABASE_NAME,
            "message": "MongoDB Atlas connection successful!"
        }

    except (PyMongoError, RuntimeError) as exc:
        return {
            "success": False,
            "message": str(exc)
        }


def close_connection():
    global _client

    if _client is not None:
        _client.close()
        _client = None
