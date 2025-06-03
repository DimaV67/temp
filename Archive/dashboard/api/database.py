# dashboard/api/database.py
import psycopg2
from psycopg2.extras import RealDictCursor
import os

def get_db():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()