import json
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


def test_save_group_stores_compliance_in_results(tmp_db):
    compliance_data = {
        "violations": ["missing_upc", "invalid_serving"],
        "per_field": {"upc": "missing", "serving_size": "invalid"}
    }
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={"brand_name": "TEST"},
        overall_pass=False,
        compliance=compliance_data,
    ))
    records = db_module.get_verifications()
    results = json.loads(records[0]["results"])
    assert results["compliance"] == compliance_data
    assert results["overall_pass"] is False


def test_save_group_uses_provided_batch_id(tmp_db):
    save_group(SaveGroupRequest(
        image_filename="label.jpg",
        extracted={},
        overall_pass=True,
        batch_id="test-batch-123",
    ))
    records = db_module.get_verifications()
    assert records[0]["batch_id"] == "test-batch-123"
