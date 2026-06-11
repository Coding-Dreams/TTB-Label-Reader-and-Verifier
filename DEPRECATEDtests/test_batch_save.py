import json
import pytest
import app.services.db as db_module
from app.routers.batch import save_group, SaveGroupRequest


def test_save_group_returns_ok(tmp_db):
    result = save_group(SaveGroupRequest(
        image_filename="COLA12Front.jpg",
        back_image_filename="COLA12Back.jpg",
        extracted={"brand_name": "TEST BRAND"},
    ))
    assert result == {"ok": True}


def test_save_group_persists_to_db(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={"brand_name": "MY BRAND"},
    ))
    records = db_module.get_verifications()
    assert len(records) == 1
    assert records[0]["image_filename"] == "label.jpg"
    # empty extracted derives overall_pass=False (required fields missing)
    assert records[0]["overall_pass"] == 0


def test_save_group_with_back_image_stores_combined_filename(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="front.jpg",
        back_image_filename="back.jpg",
        extracted={},
    ))
    records = db_module.get_verifications()
    assert records[0]["image_filename"] == "front.jpg + back.jpg"


def test_save_group_stores_compliance_in_results(tmp_db):
    compliance_data = {
        "compliant": False,
        "violations": [{"field": "brand_name", "label": "Brand name", "message": "Brand name is required on every alcohol label"}]
    }
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={"brand_name": "TEST"},
        compliance=compliance_data,
    ))
    records = db_module.get_verifications()
    results = json.loads(records[0]["results"])
    assert results["compliance"] == compliance_data
    # server-side check_compliance on {"brand_name": "TEST"} → missing other required fields → False
    assert results["overall_pass"] is False


def test_save_group_uses_provided_batch_id(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={},
        batch_id="test-batch-123",
    ))
    records = db_module.get_verifications()
    assert records[0]["batch_id"] == "test-batch-123"
