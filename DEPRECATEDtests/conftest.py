import os
import pytest
from pathlib import Path
import app.services.db as db_module


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_file)
    db_module.init_db()
    return db_file


@pytest.fixture(scope="session")
def show_ocr() -> bool:
    """True when SHOW_OCR=1 is set in the environment (passed by run_tests.sh --verbose)."""
    return os.getenv("SHOW_OCR", "0") == "1"
