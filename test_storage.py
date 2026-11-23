
from src.storage import migrate_local_records, save_case, list_cases

print("Migrating existing local records...")

migrated = migrate_local_records()
print("New records imported:", migrated)

print("\nTesting new MongoDB record...")

test_id = save_case({
    "filename": "mongodb_test.png",
    "task": "storage_connection_test",
    "predicted_class": "Test",
    "confidence": 0.99,
    "model": "StorageTest",
    "status": "test_record"
})

print("New record ID:", test_id)

print("\nReading records from MongoDB...")

records = list_cases()

print("Total records:", len(records))
print("Latest record:", records[0]["filename"])

assert any(r["case_id"] == test_id for r in records)

print("\nSUCCESS: MongoDB storage is working!")
