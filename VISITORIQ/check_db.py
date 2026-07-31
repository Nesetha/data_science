from modules.database import VisitorDatabase

db = VisitorDatabase()

for row in db.get_all_visitors():
    print(row)