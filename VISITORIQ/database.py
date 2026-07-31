import sqlite3
from datetime import datetime


class VisitorDatabase:

    def __init__(self):

        self.conn = sqlite3.connect("visitors.db")

        self.cursor = self.conn.cursor()

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                visitor_id INTEGER,
                color TEXT,
                build TEXT,
                height INTEGER,
                timestamp TEXT
            )
        """)

        self.conn.commit()

    def save_visitor(self, visitor_id, color, build, height):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.cursor.execute(
            """
            INSERT INTO visitors
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                visitor_id,
                color,
                build,
                height,
                timestamp
            )
        )

        self.conn.commit()

    def visitor_exists(self, visitor_id):

        self.cursor.execute(
            """
            SELECT * FROM visitors
            WHERE visitor_id = ?
            """,
            (visitor_id,)
        )

        result = self.cursor.fetchone()

        return result is not None

    def get_all_visitors(self):

        self.cursor.execute(
            "SELECT * FROM visitors"
        )

        return self.cursor.fetchall()