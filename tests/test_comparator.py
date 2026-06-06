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


from app.services.comparator import check_fuzzy_field


def test_fuzzy_brand_name_passes_case_variation():
    result = check_fuzzy_field("brand_name", "STONE'S THROW", "Stone's Throw", threshold=90)
    assert result.status == FieldStatus.PASS


def test_fuzzy_brand_name_passes_exact():
    result = check_fuzzy_field("brand_name", "OLD TOM DISTILLERY", "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.PASS


def test_fuzzy_brand_name_warns_on_near_miss():
    # "OLD TOM DISTELRY" scores 88 with fuzz.ratio vs "OLD TOM DISTILLERY" — in the warn range [70, 90)
    result = check_fuzzy_field("brand_name", "OLD TOM DISTELRY", "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.WARN
    assert result.score is not None
    assert 70 <= result.score < 90


def test_fuzzy_brand_name_fails_on_mismatch():
    result = check_fuzzy_field("brand_name", "BLUE RIDGE", "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.FAIL


def test_fuzzy_returns_not_detected_when_extracted_is_none():
    result = check_fuzzy_field("brand_name", None, "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.NOT_DETECTED
    assert result.extracted_value is None


def test_fuzzy_field_name_preserved():
    result = check_fuzzy_field("class_type", "Kentucky Bourbon", "Kentucky Bourbon", threshold=85)
    assert result.field == "class_type"


from app.services.comparator import check_abv


def test_abv_passes_with_full_format():
    result = check_abv("45% Alc./Vol. (90 Proof)", "45%")
    assert result.status == FieldStatus.PASS


def test_abv_passes_with_decimal():
    result = check_abv("40.0% Alc./Vol.", "40%")
    assert result.status == FieldStatus.PASS


def test_abv_fails_on_mismatch():
    result = check_abv("40%", "45%")
    assert result.status == FieldStatus.FAIL


def test_abv_passes_within_tolerance():
    result = check_abv("45.05%", "45%")
    assert result.status == FieldStatus.PASS


def test_abv_not_detected_when_none():
    result = check_abv(None, "45%")
    assert result.status == FieldStatus.NOT_DETECTED


def test_abv_field_name_is_correct():
    result = check_abv("45%", "45%")
    assert result.field == "alcohol_content"
