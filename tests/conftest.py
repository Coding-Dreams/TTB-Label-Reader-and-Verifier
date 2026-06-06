import pytest
from pathlib import Path
import app.services.db as db_module


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(db_module, "DB_PATH", db_file)
    db_module.init_db()
    return db_file
