import asyncio
import io
import logging
import os
import re
import base64
import json
import unicodedata
from typing import Optional
import httpx
import pytesseract
from pathlib import Path
from PIL import Image, ImageEnhance

from app.models.label import LabelFields

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "qwen2.5vl:7b"
TIMEOUT = 120.0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 3 prompt — semantic fields only; OCR text injected at call time
# ---------------------------------------------------------------------------

_SEMANTIC_PROMPT_TEMPLATE = """\
You are an alcohol label field extractor.

OCR text extracted from this label:
{ocr_text}

Using the OCR text above AND the label image, return ONLY valid JSON with these 4 fields:

{{
  "brand_name": "string or null",
  "class_type": "Wine" or "Malt Beverage" or "Distilled Spirits" or null,
  "producer_name_address": "string or null",
  "country_of_origin": "string or null"
}}

Rules:
- brand_name: the product/label name printed on the front (e.g. "Cascade Val", "Fete Rose", "Barenjager") — NOT the producer company name, NOT the brewery/winery/distillery name
- class_type: EXACTLY "Wine", "Malt Beverage", or "Distilled Spirits". Wine = grape/fruit wine, champagne, cider. Malt Beverage = beer, ale, lager, IPA, hard seltzer. Distilled Spirits = whiskey, bourbon, vodka, gin, rum, tequila, brandy, cognac, liqueur. If VODKA/GIN/RUM/WHISKEY appears on the label, always use "Distilled Spirits" even for canned cocktails.
- producer_name_address: the US BOTTLER, IMPORTER, or DOMESTIC PRODUCER — a company with a United States address. For imported products look for "IMPORTED BY:", "SOLE IMPORTER:", "BOTTLED BY:" followed by a US company and city/state. For domestic products return the US producer address. Return null only if truly no US entity is present.
- country_of_origin: only if the label explicitly states the product's origin country (e.g. "Product of France", "Made in Germany"). Do NOT infer from beverage style. Return null otherwise.
- Set any field to null if not found.

Return ONLY the JSON object, no markdown, no explanation.

Example for an imported Austrian wine with an NJ importer:
{{
  "brand_name": "Fete Rose",
  "class_type": "Wine",
  "producer_name_address": "Niche W. & S., CEDAR KNOLLS, NJ",
  "country_of_origin": "AUSTRIA"
}}"""

# ---------------------------------------------------------------------------
# Stage 2 regex constants
# ---------------------------------------------------------------------------

# Matches the government warning block from the keyword to end of text
_GOV_WARNING_RE = re.compile(
    r'(GOVERNMENT\s+WARNING\s*:.+)',
    re.IGNORECASE | re.DOTALL,
)

# Matches ABV statements: "35%", "13.0% ALC by VOL", "ALC. 21% BY VOL. / 42 PROOF"
_ABV_RE = re.compile(
    r'(\d+\.?\d*\s*%(?:\s*(?:alc\.?[/\s]?vol\.?|by\s+vol\.?|proof))?(?:\s*/\s*\d+\s*proof)?)',
    re.IGNORECASE,
)

# Matches volume quantities: "750ml", "1.5L", "100mL", "1 pint"
_VOLUME_RE = re.compile(
    r'(\d+\.?\d*\s*(?:ml\b|l\b|fl\.?\s*oz\b|fluid\s*oz\b|pint\b))',
    re.IGNORECASE,
)

# Matches any sulfite mention (positive or negative)
_SULFITE_RE = re.compile(
    r'((?:contains?\s+)?(?:no\s+detectable\s+)?sulfites?'
    r'|sulfiting\s+agents?'
    r'|sulfite\s+free'
    r'|no\s+sulfites?\s+added)',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Postprocessing helpers (used by Stage 3 VLM output)
# ---------------------------------------------------------------------------

_FIELD_NAMES = {
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "country_of_origin", "government_warning", "contains_sulfites",
}

_SINGLE_LINE_FIELDS = {"brand_name", "class_type", "alcohol_content", "net_contents", "country_of_origin", "contains_sulfites"}
_NULL_SENTINELS = {"none", "null", "n/a", "na", "[none]", "unknown", "-"}
_INVALID_COUNTRIES = {"american", "domestic", "imported", "local"}

_CLASS_TYPE_KEYWORDS = [
    ("wine", "Wine"), ("champagne", "Wine"), ("prosecco", "Wine"),
    ("mead", "Wine"), ("cider", "Wine"), ("vermouth", "Wine"),
    ("ale", "Malt Beverage"), ("beer", "Malt Beverage"), ("lager", "Malt Beverage"),
    ("stout", "Malt Beverage"), ("porter", "Malt Beverage"), ("ipa", "Malt Beverage"),
    ("malt", "Malt Beverage"), ("saison", "Malt Beverage"), ("pilsner", "Malt Beverage"),
    ("bock", "Malt Beverage"), ("seltzer", "Malt Beverage"),
    ("whisky", "Distilled Spirits"), ("whiskey", "Distilled Spirits"),
    ("bourbon", "Distilled Spirits"), ("scotch", "Distilled Spirits"),
    ("rum", "Distilled Spirits"), ("vodka", "Distilled Spirits"),
    ("gin", "Distilled Spirits"), ("tequila", "Distilled Spirits"),
    ("mezcal", "Distilled Spirits"), ("brandy", "Distilled Spirits"),
    ("cognac", "Distilled Spirits"), ("liqueur", "Distilled Spirits"),
    ("schnapps", "Distilled Spirits"), ("spirit", "Distilled Spirits"),
]

_COMPANY_TYPE_RE = re.compile(
    r'\b(distillery|brewery|winery|vineyard|estate)\b', re.IGNORECASE
)

_ORIGIN_PREFIX_RE = re.compile(
    r'^(?:produced?\s+in|product\s+of|made\s+in|imported?\s+from)\s+',
    re.IGNORECASE,
)


def _normalize_class_type(value: str) -> Optional[str]:
    v = value.strip().lower()
    if v in ("wine",):
        return "Wine"
    if v in ("malt beverage", "malt beverages"):
        return "Malt Beverage"
    if v in ("distilled spirits", "distilled spirit"):
        return "Distilled Spirits"
    for keyword, category in _CLASS_TYPE_KEYWORDS:
        if keyword in v:
            return category
    return None


# ---------------------------------------------------------------------------
# Stage 1: OCR
# ---------------------------------------------------------------------------

def _ocr_image(image_path: Path) -> str:
    """Run Tesseract OCR on a label image panel.

    Converts to greyscale, upscales to at least 1600px on the longest side,
    and boosts contrast before OCR. PSM 3 (auto page segmentation) handles the
    mixed layouts found on alcohol labels better than PSM 6 (uniform block).
    """
    with Image.open(image_path) as img:
        img = img.convert("L")  # greyscale — reduces noise from colour backgrounds
        w, h = img.size
        if max(w, h) < 1600:
            scale = 1600 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return pytesseract.image_to_string(img, config="--psm 3")


# ---------------------------------------------------------------------------
# Stage 2: Rule-based extraction
# ---------------------------------------------------------------------------

def _rule_extract(text: str) -> dict:
    """Extract structured fields from OCR text using deterministic rules.

    Returns only fields matched confidently. Absent keys let Stage 3 fill them.
    """
    result: dict = {}

    abv_m = _ABV_RE.search(text)
    if abv_m:
        result["alcohol_content"] = abv_m.group(1).strip()

    vol_m = _VOLUME_RE.search(text)
    if vol_m:
        result["net_contents"] = vol_m.group(1).strip()

    gw_m = _GOV_WARNING_RE.search(text)
    if gw_m:
        warning = re.sub(
            r'^government\s+warning\s*:',
            'GOVERNMENT WARNING:',
            gw_m.group(1).strip(),
            flags=re.IGNORECASE,
        )
        result["government_warning"] = warning.strip()

    sf_m = _SULFITE_RE.search(text)
    if sf_m:
        result["contains_sulfites"] = sf_m.group(0).strip()

    text_lower = text.lower()
    for keyword, category in _CLASS_TYPE_KEYWORDS:
        if keyword in text_lower:
            result["class_type"] = category
            break

    return result


# ---------------------------------------------------------------------------
# Stage 3: VLM encoding + call
# ---------------------------------------------------------------------------

def _encode_image(image_path: Path, max_side: int = 768, enhance: bool = False) -> str:
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        if enhance:
            img = ImageEnhance.Contrast(img).enhance(1.8)
            img = ImageEnhance.Sharpness(img).enhance(1.5)
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


async def _warmup_model() -> None:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": 0.1},
                },
            )
        logger.info("Model warmed up and loaded into VRAM")
    except Exception as e:
        logger.warning(f"Model warmup failed (will load on first request): {e}")


async def _call_ollama(client: httpx.AsyncClient, image_b64: str, prompt: str) -> str:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        },
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def _postprocess(data: dict) -> dict:
    # Flatten list values — take the first non-empty string element
    for key in list(data.keys()):
        if isinstance(data[key], list):
            data[key] = next((str(v).strip() for v in data[key] if v), None)

    # Normalize Unicode fullwidth characters and empty strings to None
    for key in list(data.keys()):
        if isinstance(data[key], str):
            data[key] = unicodedata.normalize("NFKC", data[key]).strip() or None

    # For single-line fields, discard everything after the first newline
    for key in _SINGLE_LINE_FIELDS:
        val = data.get(key)
        if isinstance(val, str) and "\n" in val:
            data[key] = val.split("\n")[0].strip() or None

    # Null out sentinel/garbage values
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, str) and val.strip().lower() in _NULL_SENTINELS:
            data[key] = None

    # Null out values that are JSON key names (model confused structure with content)
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, str) and val.strip().lower().replace(" ", "_") in _FIELD_NAMES:
            data[key] = None

    # Strip leading origin phrases from country_of_origin
    country = data.get("country_of_origin")
    if isinstance(country, str):
        stripped = _ORIGIN_PREFIX_RE.sub("", country.strip()).strip()
        if stripped != country.strip():
            data["country_of_origin"] = stripped
            country = stripped

    # Null out country_of_origin if it's not an actual country name
    country = data.get("country_of_origin")
    if isinstance(country, str):
        if re.fullmatch(r"[A-Z]{2}", country.strip()):
            data["country_of_origin"] = None
        elif country.strip().lower() in _INVALID_COUNTRIES:
            data["country_of_origin"] = None
        else:
            country_lower = country.strip().lower()
            class_type = (data.get("class_type") or "").lower()
            producer = (data.get("producer_name_address") or "").lower()
            if country_lower in ("united states", "america") and (
                "american" in class_type or "domestic" in class_type
            ):
                data["country_of_origin"] = None
            elif country_lower in ("united states", "america", "usa", "u.s.", "u.s.a.") and (
                "import" in producer
            ):
                data["country_of_origin"] = None

    # Ensure government_warning includes the required prefix if VLM returned it
    gw = data.get("government_warning")
    if gw and not gw.upper().lstrip().startswith("GOVERNMENT WARNING"):
        data["government_warning"] = "GOVERNMENT WARNING: " + gw.strip()

    # Null out brand_name if it's actually the producer name
    if data.get("brand_name") and data.get("producer_name_address"):
        bn_lower = data["brand_name"].strip().lower()
        prod_lower = data["producer_name_address"].strip().lower()
        if _COMPANY_TYPE_RE.search(data["brand_name"]) and bn_lower in prod_lower:
            data["brand_name"] = None

    # Normalize class_type to one of three canonical categories
    ct = data.get("class_type")
    if isinstance(ct, str):
        data["class_type"] = _normalize_class_type(ct)

    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_label_fields(
    image_path: Path, back_image_path: Optional[Path] = None
) -> LabelFields:
    loop = asyncio.get_running_loop()

    # Stage 1 — OCR both panels at full resolution
    try:
        front_text = await loop.run_in_executor(None, _ocr_image, image_path)
        back_text = ""
        if back_image_path and back_image_path.exists():
            back_text = await loop.run_in_executor(None, _ocr_image, back_image_path)
        combined_text = (front_text + "\n\n" + back_text).strip()
    except Exception as e:
        logger.warning("OCR failed, falling back to VLM-only: %s", e)
        combined_text = ""

    # Stage 2 — Rule-based extraction from OCR text
    rule_data = _rule_extract(combined_text) if combined_text else {}

    # Stage 3 — VLM for semantic fields (brand, class, producer, country)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        front_b64 = await loop.run_in_executor(None, _encode_image, image_path, 1024)
        ocr_context = combined_text[:3000] if combined_text else "(OCR unavailable — read from image)"
        prompt = _SEMANTIC_PROMPT_TEMPLATE.format(ocr_text=ocr_context)
        raw = await _call_ollama(client, front_b64, prompt)
        vlm_data = _postprocess(_parse_json(raw))

    # Merge: Stage 2 (rules) takes precedence for structured fields;
    # Stage 3 (VLM) fills semantic fields rules cannot handle.
    merged: dict = {
        # Semantic fields — VLM only
        "brand_name": vlm_data.get("brand_name"),
        "producer_name_address": vlm_data.get("producer_name_address"),
        "country_of_origin": vlm_data.get("country_of_origin"),
        # class_type — VLM preferred; keyword match as fallback
        "class_type": vlm_data.get("class_type") or rule_data.get("class_type"),
        # Structured fields — rules take precedence; VLM as fallback
        "alcohol_content": rule_data.get("alcohol_content") or vlm_data.get("alcohol_content"),
        "net_contents": rule_data.get("net_contents") or vlm_data.get("net_contents"),
        "government_warning": rule_data.get("government_warning") or vlm_data.get("government_warning"),
        "contains_sulfites": rule_data.get("contains_sulfites") or vlm_data.get("contains_sulfites"),
    }

    return LabelFields(**merged)
