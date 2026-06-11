import json
import pytest
from app.services.db import save_verification, get_verifications, get_verification


def test_save_and_retrieve_verification(tmp_db):
    form_data = {"brand_name": "OLD TOM", "government_warning": "GOVERNMENT WARNING: ..."}
    extracted = {"brand_name": "OLD TOM DISTILLERY", "government_warning": "GOVERNMENT WARNING: ..."}
    results = {"overall_pass": True, "fields": []}

    record_id = save_verification(
        image_filename="label.jpg",
        form_data=form_data,
        extracted=extracted,
        results=results,
        overall_pass=True,
    )

    assert record_id == 1
    record = get_verification(record_id)
    assert record["image_filename"] == "label.jpg"
    assert record["overall_pass"] == 1
    assert json.loads(record["form_data"])["brand_name"] == "OLD TOM"


def test_get_verifications_returns_list(tmp_db):
    save_verification("a.jpg", {}, {}, {}, True)
    save_verification("b.jpg", {}, {}, {}, False)

    records = get_verifications()
    assert len(records) == 2
    assert records[0]["image_filename"] == "b.jpg"  # Most recent first


def test_get_verification_returns_none_for_missing(tmp_db):
    assert get_verification(999) is None


def test_batch_id_stored(tmp_db):
    save_verification("label.jpg", {}, {}, {}, True, batch_id="batch-abc")
    records = get_verifications()
    assert records[0]["batch_id"] == "batch-abc"
