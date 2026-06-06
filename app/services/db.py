import sqlite3
import json
from pathlib import Path
from typing import Optional, List

DB_PATH = Path("data/verifications.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     DATETIME DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
                image_filename TEXT NOT NULL,
                form_data      TEXT NOT NULL,
                extracted      TEXT NOT NULL,
                results        TEXT NOT NULL,
                overall_pass   BOOLEAN NOT NULL,
                batch_id       TEXT
            )
        """)
        conn.commit()


def save_verification(
    image_filename: str,
    form_data: dict,
    extracted: dict,
    results: dict,
    overall_pass: bool,
    batch_id: Optional[str] = None,
) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """INSERT INTO verifications
               (image_filename, form_data, extracted, results, overall_pass, batch_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                image_filename,
                json.dumps(form_data),
                json.dumps(extracted),
                json.dumps(results),
                overall_pass,
                batch_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_verifications(limit: int = 100) -> List[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM verifications ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_verification(verification_id: int) -> Optional[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM verifications WHERE id = ?", (verification_id,)
        ).fetchone()
        return dict(row) if row else None
