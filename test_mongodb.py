
from src.database import test_connection

result = test_connection()

if result["success"]:
    print("SUCCESS!")
    print(result["message"])
    print("Database:", result["database"])
else:
    print("CONNECTION FAILED!")
    print(result["message"])
