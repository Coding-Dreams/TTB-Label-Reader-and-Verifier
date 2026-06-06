import pytest
from app.models.result import FieldStatus
from app.services.comparator import check_government_warning


def test_government_warning_passes_with_all_caps():
    result = check_government_warning(
        extracted="GOVERNMENT WARNING: According to the Surgeon General...",
        submitted="GOVERNMENT WARNING: According to the Surgeon General...",
    )
    assert result.status == FieldStatus.PASS


def test_government_warning_fails_with_title_case():
    result = check_government_warning(
        extracted="Government Warning: According to the Surgeon General...",
        submitted="GOVERNMENT WARNING: According to the Surgeon General...",
    )
    assert result.status == FieldStatus.FAIL


def test_government_warning_fails_when_missing_from_label():
    result = check_government_warning(extracted=None, submitted="GOVERNMENT WARNING: ...")
    assert result.status == FieldStatus.NOT_DETECTED


def test_government_warning_field_name_is_correct():
    result = check_government_warning(extracted="GOVERNMENT WARNING: ...", submitted=None)
    assert result.field == "government_warning"


def test_government_warning_fails_when_substring_absent():
    result = check_government_warning(
        extracted="Warning: Drinking is bad for you.",
        submitted="GOVERNMENT WARNING: ...",
    )
    assert result.status == FieldStatus.FAIL
