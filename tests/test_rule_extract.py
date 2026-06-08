"""Unit tests for _rule_extract() — Stage 2 deterministic field extraction."""
import pytest
from app.services.ollama import _rule_extract


# --- alcohol_content ---

def test_rule_extract_abv_simple():
    result = _rule_extract("ALCOHOL CONTENT: 35%")
    assert result.get("alcohol_content") is not None
    assert "35" in result["alcohol_content"]


def test_rule_extract_abv_with_qualifier():
    result = _rule_extract("61% ALC./VOL (122 PROOF)")
    assert result.get("alcohol_content") is not None
    assert "61" in result["alcohol_content"]


def test_rule_extract_abv_by_vol():
    result = _rule_extract("ALC. 21% BY VOL. / 42 PROOF")
    assert result.get("alcohol_content") is not None
    assert "21" in result["alcohol_content"]


def test_rule_extract_abv_decimal():
    result = _rule_extract("13.0% ALC by VOL")
    assert result.get("alcohol_content") is not None
    assert "13" in result["alcohol_content"]


def test_rule_extract_abv_alc_vol():
    result = _rule_extract("8% ALC/VOL")
    assert result.get("alcohol_content") is not None
    assert "8" in result["alcohol_content"]


# --- net_contents ---

def test_rule_extract_volume_ml_uppercase():
    result = _rule_extract("NET CONTENTS: 750ML")
    assert result.get("net_contents") is not None
    assert "750" in result["net_contents"]


def test_rule_extract_volume_litre():
    result = _rule_extract("1.5L")
    assert result.get("net_contents") is not None
    assert "1.5" in result["net_contents"]


def test_rule_extract_volume_lowercase():
    result = _rule_extract("100ml net contents")
    assert result.get("net_contents") is not None
    assert "100" in result["net_contents"]


def test_rule_extract_volume_375():
    result = _rule_extract("NET CONTENTS 375ML")
    assert result.get("net_contents") is not None
    assert "375" in result["net_contents"]


# --- government_warning ---

def test_rule_extract_government_warning_found():
    text = (
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
        "alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems."
    )
    result = _rule_extract(text)
    assert result.get("government_warning") is not None
    assert result["government_warning"].startswith("GOVERNMENT WARNING:")


def test_rule_extract_government_warning_normalises_case():
    text = "government warning: (1) According to the Surgeon General..."
    result = _rule_extract(text)
    assert result.get("government_warning") is not None
    assert result["government_warning"].startswith("GOVERNMENT WARNING:")


def test_rule_extract_government_warning_absent():
    result = _rule_extract("Brand Name: Cascade Val. Alcohol: 11.5%")
    assert result.get("government_warning") is None


# --- contains_sulfites ---

def test_rule_extract_sulfites_present():
    result = _rule_extract("CONTAINS SULFITES")
    assert result.get("contains_sulfites") is not None
    assert "sulfite" in result["contains_sulfites"].lower()


def test_rule_extract_sulfites_free():
    result = _rule_extract("SULFITE FREE")
    assert result.get("contains_sulfites") is not None
    assert "sulfite" in result["contains_sulfites"].lower()


def test_rule_extract_sulfites_no_detectable():
    result = _rule_extract("Contains No Detectable Sulfites")
    assert result.get("contains_sulfites") is not None
    assert "sulfite" in result["contains_sulfites"].lower()


def test_rule_extract_sulfites_absent():
    result = _rule_extract("Distilled Spirits, 750ml, 40% ALC/VOL")
    assert result.get("contains_sulfites") is None


# --- class_type ---

def test_rule_extract_class_wine():
    result = _rule_extract("California Cabernet Sauvignon Wine")
    assert result.get("class_type") == "Wine"


def test_rule_extract_class_malt():
    result = _rule_extract("India Pale Ale brewed and bottled by")
    assert result.get("class_type") == "Malt Beverage"


def test_rule_extract_class_spirits_whiskey():
    result = _rule_extract("Kentucky Straight Bourbon Whiskey")
    assert result.get("class_type") == "Distilled Spirits"


def test_rule_extract_class_spirits_vodka():
    result = _rule_extract("Premium Vodka distilled from grain")
    assert result.get("class_type") == "Distilled Spirits"


def test_rule_extract_class_absent():
    # No beverage-type keyword — VLM should handle it
    result = _rule_extract("Product of France 750ML IMPORTED BY: Acme Co., New York, NY")
    assert result.get("class_type") is None


# --- combined real-world-ish label text ---

def test_rule_extract_combined_wine_label():
    text = (
        "Fete Rose\n"
        "12% ALC/VOL\n"
        "750ML\n"
        "CONTAINS SULFITES\n"
        "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink "
        "alcoholic beverages during pregnancy because of the risk of birth defects. "
        "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
        "operate machinery, and may cause health problems."
    )
    result = _rule_extract(text)
    assert "12" in result.get("alcohol_content", "")
    assert "750" in result.get("net_contents", "")
    assert result.get("contains_sulfites") is not None
    assert result.get("government_warning", "").startswith("GOVERNMENT WARNING:")


# --- edge cases ---

def test_rule_extract_empty_string():
    assert _rule_extract("") == {}


def test_rule_extract_no_matches():
    assert _rule_extract("Lorem ipsum dolor sit amet") == {}
