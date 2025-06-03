import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from scripts.utils import load_config


class BrandscopeDB:
    def __init__(self, db_path: str = None):
        """Initialize SQLite database connection."""
        config = load_config()
        self.db_path = Path(db_path or config['file_paths']['db_path'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        """Create tables for users, companies, brands, products, user_company_access, and llm_results."""
        # Users table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Companies table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                url TEXT NOT NULL,
                industry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Brands table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS brands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                industry TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies(id),
                UNIQUE (company_id, name)
            )
        """)

        # Products table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (brand_id) REFERENCES brands(id),
                UNIQUE (brand_id, name)
            )
        """)

        # User_company_access table for user access to companies
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_company_access (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                company_id INTEGER NOT NULL,
                access_level TEXT NOT NULL,  -- e.g., 'read', 'write', 'admin'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (company_id) REFERENCES companies(id),
                UNIQUE (user_id, company_id)
            )
        """)

        # LLM results table (retained for compatibility)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS llm_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                brand_id INTEGER NOT NULL,
                accuracy REAL,
                claims_extracted INTEGER,
                interim_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (brand_id) REFERENCES brands(id)
            )
        """)
        self.conn.commit()

    def insert_user(self, username: str, email: str) -> int:
        """Insert a new user and return their ID."""
        try:
            self.cursor.execute("""
                INSERT INTO users (username, email)
                VALUES (?, ?)
            """, (username, email))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("Username or email already exists")

    def insert_company(self, name: str, url: str, industry: str = None) -> int:
        """Insert a company and return its ID."""
        try:
            self.cursor.execute("""
                INSERT INTO companies (name, url, industry)
                VALUES (?, ?, ?)
            """, (name, url, industry))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("Company name already exists")

    def insert_brand(self, company_id: int, name: str, url: str, industry: str = None) -> int:
        """Insert a brand for a company and return its ID."""
        try:
            self.cursor.execute("""
                INSERT INTO brands (company_id, name, url, industry)
                VALUES (?, ?, ?, ?)
            """, (company_id, name, url, industry))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("Brand name already exists for this company")

    def insert_product(self, brand_id: int, name: str, url: str) -> int:
        """Insert a product for a brand and return its ID."""
        try:
            self.cursor.execute("""
                INSERT INTO products (brand_id, name, url)
                VALUES (?, ?, ?)
            """, (brand_id, name, url))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError("Product name already exists for this brand")

    def insert_user_company_access(self, user_id: int, company_id: int, access_level: str):
        """Insert user access to a company."""
        try:
            self.cursor.execute("""
                INSERT INTO user_company_access (user_id, company_id, access_level)
                VALUES (?, ?, ?)
            """, (user_id, company_id, access_level))
            self.conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError("User already has access to this company")

    def insert_llm_result(self, result: Dict[str, Any]):
        """Insert an LLM evaluation result."""
        self.cursor.execute("""
            INSERT INTO llm_results (model_name, brand_id, accuracy, claims_extracted, interim_file)
            VALUES (?, ?, ?, ?, ?)
        """, (
            result.get("model_name", ""),
            result.get("brand_id", 0),
            result.get("accuracy", 0.0),
            result.get("claims_extracted", 0),
            result.get("interim_file", "")
        ))
        self.conn.commit()

    def query_companies(self, user_id: int) -> List[Dict[str, Any]]:
        """Query companies a user has access to."""
        self.cursor.execute("""
            SELECT c.* FROM companies c
            JOIN user_company_access uca ON c.id = uca.company_id
            WHERE uca.user_id = ?
        """, (user_id,))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def query_brands(self, company_id: int) -> List[Dict[str, Any]]:
        """Query brands for a company."""
        self.cursor.execute("SELECT * FROM brands WHERE company_id = ?", (company_id,))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def query_products(self, brand_id: int) -> List[Dict[str, Any]]:
        """Query products for a brand."""
        self.cursor.execute("SELECT * FROM products WHERE brand_id = ?", (brand_id,))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def query_llm_results(self, brand_id: int) -> List[Dict[str, Any]]:
        """Query LLM results for a brand."""
        self.cursor.execute("SELECT * FROM llm_results WHERE brand_id = ?", (brand_id,))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def query_user_access(self, user_id: int) -> List[Dict[str, Any]]:
        """Query user access to companies."""
        self.cursor.execute("SELECT * FROM user_company_access WHERE user_id = ?", (user_id,))
        columns = [desc[0] for desc in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]

    def close(self):
        """Close database connection."""
        self.conn.close()

if __name__ == "__main__":
    # Example usage
    db = BrandscopeDB()
    
    # Add a user
    user_id = db.insert_user("john_doe", "john@example.com")
    
    # Add a company
    company_id = db.insert_company("Maelove Inc.", "https://maelove.com", "Skincare")
    
    # Grant user access to company
    db.insert_user_company_access(user_id, company_id, "write")
    
    # Add a brand
    brand_id = db.insert_brand(company_id, "Maelove Skincare", "https://maelove.com", "Skincare")
    
    # Add a product
    db.insert_product(brand_id, "Glow Maker Vitamin C Serum", "https://maelove.com/products/glow-maker")
    
    # Add an LLM result
    sample_result = {
        "model_name": "ollama-llama3.2-latest",
        "brand_id": brand_id,
        "accuracy": 0.85,
        "claims_extracted": 10,
        "interim_file": "maelove-skincare-ollama-llama3.2-latest-interim-20250411_170644.json"
    }
    db.insert_llm_result(sample_result)
    
    # Query data
    print(db.query_companies(user_id))
    print(db.query_brands(company_id))
    print(db.query_products(brand_id))
    print(db.query_llm_results(brand_id))
    print(db.query_user_access(user_id))
    
    db.close()