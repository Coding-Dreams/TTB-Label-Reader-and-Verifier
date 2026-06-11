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
    assert result.status == FieldStatus.FAIL


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


def test_fuzzy_fails_when_extracted_is_none():
    result = check_fuzzy_field("brand_name", None, "OLD TOM DISTILLERY", threshold=90)
    assert result.status == FieldStatus.FAIL
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


def test_abv_fails_when_extracted_is_none():
    result = check_abv(None, "45%")
    assert result.status == FieldStatus.FAIL


def test_abv_field_name_is_correct():
    result = check_abv("45%", "45%")
    assert result.field == "alcohol_content"


from app.services.comparator import check_net_contents, check_exact_field, check_sulfites, compare_label
from app.models.label import LabelFields


def test_net_contents_passes_case_insensitive_ml():
    result = check_net_contents("750 mL", "750ml")
    assert result.status == FieldStatus.PASS


def test_net_contents_passes_litre_to_ml():
    result = check_net_contents("0.75 L", "750 mL")
    assert result.status == FieldStatus.PASS


def test_net_contents_fails_different_volume():
    result = check_net_contents("750 mL", "1000 mL")
    assert result.status == FieldStatus.FAIL


def test_net_contents_fails_when_extracted_is_none():
    result = check_net_contents(None, "750 mL")
    assert result.status == FieldStatus.FAIL


def test_exact_field_passes_case_insensitive():
    result = check_exact_field("country_of_origin", "united states", "United States")
    assert result.status == FieldStatus.PASS


def test_exact_field_fails_mismatch():
    result = check_exact_field("country_of_origin", "France", "United States")
    assert result.status == FieldStatus.FAIL


def test_exact_field_fails_when_extracted_is_none():
    result = check_exact_field("country_of_origin", None, "United States")
    assert result.status == FieldStatus.FAIL


def test_sulfites_passes_when_both_declare():
    result = check_sulfites("CONTAINS SULFITES", "Contains Sulfites")
    assert result.status == FieldStatus.PASS
    assert result.field == "contains_sulfites"


def test_sulfites_fails_when_label_missing():
    result = check_sulfites(None, "Contains Sulfites")
    assert result.status == FieldStatus.FAIL


def test_sulfites_not_detected_when_submitted_is_none():
    # submitted=None means "not provided" — treated as NOT_DETECTED, not a FAIL
    result = check_sulfites("CONTAINS SULFITES", None)
    assert result.status == FieldStatus.NOT_DETECTED


def test_sulfites_not_detected_when_neither_declares():
    result = check_sulfites(None, None)
    assert result.status == FieldStatus.NOT_DETECTED


def test_sulfites_passes_when_both_declare_free():
    result = check_sulfites("SULFITE FREE", "SULFITE FREE")
    assert result.status == FieldStatus.PASS


def test_sulfites_fails_when_positive_vs_negative_mismatch():
    result = check_sulfites("CONTAINS SULFITES", "SULFITE FREE")
    assert result.status == FieldStatus.FAIL


def test_sulfites_fails_when_negative_vs_positive_mismatch():
    result = check_sulfites("SULFITE FREE", "CONTAINS SULFITES")
    assert result.status == FieldStatus.FAIL


def test_sulfites_not_detected_when_label_is_free_but_submission_omits():
    result = check_sulfites("SULFITE FREE", None)
    assert result.status == FieldStatus.NOT_DETECTED


def test_compare_label_all_pass():
    warning = "GOVERNMENT WARNING: According to the Surgeon General, women should not drink alcoholic beverages during pregnancy because of the risk of birth defects."
    extracted = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45% Alc./Vol. (90 Proof)",
        net_contents="750 mL",
        producer_name_address="Old Tom Distillery, Louisville, KY",
        country_of_origin="United States",
        government_warning=warning,
        contains_sulfites=None,
    )
    form_data = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        class_type="Kentucky Straight Bourbon Whiskey",
        alcohol_content="45%",
        net_contents="750ml",
        producer_name_address="Old Tom Distillery, Louisville, KY",
        country_of_origin="United States",
        government_warning=warning,
        contains_sulfites=None,
    )
    result = compare_label(extracted, form_data)
    assert result.overall_pass is True
    # contains_sulfites is NOT_DETECTED (no sulfite declaration on label — expected for spirits)
    assert all(f.status != FieldStatus.FAIL for f in result.fields)


def test_compare_label_fails_on_bad_warning():
    extracted = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        government_warning="Government Warning: (lowercase)",
    )
    form_data = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        government_warning="GOVERNMENT WARNING: ...",
    )
    result = compare_label(extracted, form_data)
    assert result.overall_pass is False
    warning_result = next(f for f in result.fields if f.field == "government_warning")
    assert warning_result.status == FieldStatus.FAIL


def test_compare_label_warn_does_not_fail_overall():
    warning = "GOVERNMENT WARNING: ..."
    extracted = LabelFields(
        brand_name="OLD TOM DISTELRY",  # typo — should be WARN not FAIL
        government_warning=warning,
    )
    form_data = LabelFields(
        brand_name="OLD TOM DISTILLERY",
        government_warning=warning,
    )
    result = compare_label(extracted, form_data)
    brand_result = next(f for f in result.fields if f.field == "brand_name")
    assert brand_result.status != FieldStatus.FAIL  # near-miss should not fail overall
    assert result.overall_pass is True
