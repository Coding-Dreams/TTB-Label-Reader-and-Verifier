import pytest
import app.services.db as db_module
from app.routers.batch import save_group, SaveGroupRequest


def test_save_group_returns_ok(tmp_db):
    result = save_group(SaveGroupRequest(
        image_filename="COLA12Front.jpg",
        back_image_filename="COLA12Back.jpg",
        extracted={"brand_name": "TEST BRAND"},
        overall_pass=True,
    ))
    assert result == {"ok": True}


def test_save_group_persists_to_db(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={"brand_name": "MY BRAND"},
        overall_pass=False,
    ))
    records = db_module.get_verifications()
    assert len(records) == 1
    assert records[0]["image_filename"] == "label.jpg"
    assert records[0]["overall_pass"] == 0


def test_save_group_with_back_image_stores_combined_filename(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="front.jpg",
        back_image_filename="back.jpg",
        extracted={},
        overall_pass=True,
    ))
    records = db_module.get_verifications()
    assert records[0]["image_filename"] == "front.jpg + back.jpg"
