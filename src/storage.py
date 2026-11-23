
import json
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from src.database import get_database

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "local_cases.json"

COLLECTION_NAME = "research_cases"


def _get_collection():
    db = get_database()
    collection = db[COLLECTION_NAME]

    # Ensure every research record has a unique case ID
    collection.create_index(
        [("case_id", ASCENDING)],
        unique=True
    )

    return collection


def _read_local():
    """Read existing local JSON records without deleting them."""
    if not STORE.exists():
        return []

    try:
        return json.loads(
            STORE.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return []


def migrate_local_records():
    """Import existing JSON records into MongoDB without duplicates."""
    collection = _get_collection()
    local_records = _read_local()

    migrated = 0

    for record in local_records:
        if not record.get("case_id"):
            continue

        result = collection.update_one(
            {"case_id": record["case_id"]},
            {"$setOnInsert": record},
            upsert=True
        )

        if result.upserted_id is not None:
            migrated += 1

    return migrated


def save_case(data: dict) -> str:
    """Save a new research classification record in MongoDB."""

    migrate_local_records()

    case_id = str(uuid4())[:8]

    record = {
        "case_id": case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **data
    }

    try:
        collection = _get_collection()
        collection.insert_one(record)
        return case_id

    except PyMongoError as exc:
        raise RuntimeError(
            f"Could not save research record to MongoDB: {exc}"
        ) from exc


def list_cases():
    """Retrieve research records from MongoDB."""

    migrate_local_records()

    try:
        collection = _get_collection()

        records = list(
            collection.find(
                {},
                {"_id": 0}
            ).sort("created_at", DESCENDING)
        )

        return records

    except PyMongoError as exc:
        raise RuntimeError(
            f"Could not retrieve research records: {exc}"
        ) from exc
