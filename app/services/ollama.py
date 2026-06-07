import asyncio
import io
import logging
import os
import re
import base64
import json
import tempfile
import unicodedata
from typing import Optional
import httpx
from pathlib import Path
from PIL import Image

from app.models.label import LabelFields

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "qwen2.5vl:7b"
TIMEOUT = 120.0

_MAX_SIDE = 768  # cap large uploads before encoding

logger = logging.getLogger(__name__)

_FULL_PROMPT = """You are an OCR assistant specialized in reading alcohol beverage labels.
Return ONLY valid JSON with no additional text. Set any field to null if not visible.

Required JSON format:
{
  "brand_name": "string or null",
  "class_type": "Wine" or "Malt Beverage" or "Distilled Spirits" or null,
  "alcohol_content": "string or null",
  "net_contents": "string or null",
  "contains_sulfites": "string or null",
  "producer_name_address": "string or null",
  "country_of_origin": "string or null",
  "government_warning": "string or null"
}

Rules:
- Return the exact text as it appears on the label for all fields except class_type
- brand_name: the label/product name printed on the front that identifies this specific product (e.g. "ABC Single Barrel", "Honey Huckleberry Pie", "12345 Imports") — often the most prominent or stylistic name; do NOT capture the producer, brewery, winery, or distillery company name
- class_type: EXACTLY one of three values — "Wine", "Malt Beverage", or "Distilled Spirits". Wine = grape/fruit wines, champagne, prosecco, cider. Malt Beverage = beer, ale, lager, stout, porter, IPA, hard seltzer. Distilled Spirits = whiskey, bourbon, rum, vodka, gin, tequila, brandy, liqueur, and similar spirits. IMPORTANT: if the label shows any distilled spirit name (VODKA, GIN, RUM, WHISKEY, TEQUILA, etc.) classify as "Distilled Spirits" even if it is a flavored or canned cocktail — only use "Malt Beverage" if no distilled spirit name is present
- alcohol_content: the ABV percentage as printed (e.g. "45% ALC/VOL", "13% BY VOL")
- net_contents: the TOTAL container size (e.g. "750 ML", "100 mL", "1 PINT") — the full bottle/can volume, NOT the alcohol-per-serving amount
- contains_sulfites: search ALL panels for any sulfite statement — this includes BOTH positive declarations (e.g. "CONTAINS SULFITES", "Contains Sulfating Agents") AND negative declarations (e.g. "SULFITE FREE", "NO SULFITES ADDED", "Contains No Detectable Sulfites"); return the exact text if found, null if absent
- producer_name_address: the COMPLETE producer/bottler/importer entry as printed — capture BOTH the company name AND the full location (city, state/country) as one value (e.g. "ABC DISTILLERY FREDERICK, MD", "IMPORTED BY: 12345 IMPORTS MIAMI, FL"); do NOT return just the name or just the address alone; if BOTH a foreign producer AND a US importer/bottler/distributor are listed, return the IMPORTER/BOTTLER/DISTRIBUTOR entry — NOT the foreign producer
- country_of_origin: the country name, but ONLY if explicitly stated as the product's origin (e.g. "Product of Canada", "Made in Germany", "Imported from France"). Do NOT infer from the beverage category or style name — "American Red Wine" does NOT mean country_of_origin is "United States"
- government_warning: the COMPLETE warning text EXACTLY as printed, including the "GOVERNMENT WARNING:" heading if present"""


def _encode_image(image_path: Path, max_side: int = _MAX_SIDE) -> str:
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


def _stitch_images(front_path: Path, back_path: Path) -> Path:
    """Stitch front (left) and back (right) label images side by side at matching height."""
    with Image.open(front_path) as front_img, Image.open(back_path) as back_img:
        front_rgb = front_img.convert("RGB")
        back_rgb = back_img.convert("RGB")

        target_h = max(front_rgb.height, back_rgb.height)
        fw = int(front_rgb.width * target_h / front_rgb.height)
        bw = int(back_rgb.width * target_h / back_rgb.height)
        front_r = front_rgb.resize((fw, target_h), Image.LANCZOS)
        back_r = back_rgb.resize((bw, target_h), Image.LANCZOS)

        combined = Image.new("RGB", (fw + bw, target_h), (255, 255, 255))
        combined.paste(front_r, (0, 0))
        combined.paste(back_r, (fw, 0))

        fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="stitched_")
        os.close(fd)
        combined.save(tmp, format="JPEG", quality=95)
        return Path(tmp)


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


async def extract_label_fields(
    image_path: Path, back_image_path: Optional[Path] = None
) -> LabelFields:
    stitched: Optional[Path] = None
    try:
        back_b64: Optional[str] = None
        if back_image_path and back_image_path.exists():
            loop = asyncio.get_event_loop()
            stitched = await loop.run_in_executor(
                None, _stitch_images, image_path, back_image_path
            )
            effective_path = stitched
            # Encode back panel separately at full resolution for targeted second-pass lookups
            back_b64 = await loop.run_in_executor(None, _encode_image, back_image_path)
        else:
            effective_path = image_path

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            loop = asyncio.get_event_loop()
            image_b64 = await loop.run_in_executor(None, _encode_image, effective_path)
            raw = await _call_ollama(client, image_b64, _FULL_PROMPT)
            data = _postprocess(_parse_json(raw))
            # Back panel often carries sulfite statements and regulatory text
            sulfite_b64 = back_b64 or image_b64
            if not data.get("contains_sulfites"):
                data["contains_sulfites"] = await _extract_sulfites(client, sulfite_b64)
            if not data.get("brand_name"):
                data["brand_name"] = await _extract_brand_name(client, image_b64)
            if not data.get("class_type"):
                data["class_type"] = await _extract_class_type(client, image_b64)
            # If producer looks foreign (no US state at end), search the back panel for a US importer
            importer_b64 = back_b64 or image_b64
            producer = data.get("producer_name_address")
            if isinstance(producer, str) and not _US_ADDRESS_TAIL_RE.search(producer):
                importer = await _extract_importer(client, importer_b64)
                if importer:
                    data["producer_name_address"] = importer
            # Wine default: TTB requires sulfite declaration for wines >=10 ppm; if no
            # statement found (neither positive nor negative), assume CONTAINS SULFITES
            if not data.get("contains_sulfites") and data.get("class_type") == "Wine":
                data["contains_sulfites"] = "CONTAINS SULFITES"
            return LabelFields(**data)
    finally:
        if stitched:
            stitched.unlink(missing_ok=True)


async def _extract_brand_name(client: httpx.AsyncClient, image_b64: str) -> Optional[str]:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    "Look at this alcohol beverage label image. "
                    "What is the PRODUCT NAME or LABEL NAME of this specific beverage? "
                    "Examples: 'ABC Single Barrel', 'Honey Huckleberry Pie', '12345 Imports'. "
                    "IMPORTANT: if you see text like 'DISTILLED BY XYZ Distillery' or "
                    "'BREWED & BOTTLED BY XYZ Brewery', that is the PRODUCER name — do NOT "
                    "return it. Return the product/label name only, or 'none' if you truly "
                    "cannot identify one."
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        },
    )
    resp.raise_for_status()
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present", "cannot determine"):
        return None
    return result or None


async def _extract_class_type(client: httpx.AsyncClient, image_b64: str) -> Optional[str]:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    "Look at this alcohol beverage label. "
                    "Classify this product as EXACTLY one of three categories: "
                    "'Wine', 'Malt Beverage', or 'Distilled Spirits'. "
                    "Wine = grape or fruit wine, champagne, prosecco, cider, mead. "
                    "Malt Beverage = beer, ale, lager, stout, porter, IPA, hard seltzer, or any malt-based drink. "
                    "Distilled Spirits = whiskey, bourbon, rye, rum, vodka, gin, tequila, brandy, cognac, liqueur, or any distilled spirit. "
                    "IMPORTANT: If the label shows the word 'VODKA', 'GIN', 'RUM', 'WHISKEY', 'TEQUILA', or any other distilled spirit name — "
                    "classify as 'Distilled Spirits' even if the product is a flavored cocktail, mixed drink, or comes in a can or pouch. "
                    "Only classify as 'Malt Beverage' if the label explicitly says beer, ale, lager, brewed, or malt-based with NO distilled spirit name present. "
                    "Reply with ONLY the category name, nothing else."
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        },
    )
    resp.raise_for_status()
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if result.lower() in _NULL_SENTINELS:
        return None
    return _normalize_class_type(result) or result or None


async def _extract_sulfites(client: httpx.AsyncClient, image_b64: str) -> Optional[str]:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    'Look at this alcohol label image carefully. '
                    'Search every panel for any text about sulfites — including BOTH '
                    'positive statements like "CONTAINS SULFITES", "Contains Sulfating Agents" '
                    'AND negative statements like "SULFITE FREE", "NO SULFITES ADDED", '
                    '"Contains No Detectable Sulfites". '
                    'Reply with just that exact text if you find it, or reply with '
                    'the single word "none" if no sulfite statement is present.'
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        },
    )
    resp.raise_for_status()
    result = resp.json()["message"]["content"].strip()
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present", "absent"):
        return None
    return result


async def _extract_importer(client: httpx.AsyncClient, image_b64: str) -> Optional[str]:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    "Look at this alcohol beverage label. "
                    "Find any US IMPORTER, BOTTLER, or DISTRIBUTOR — a company located inside the United States. "
                    "IGNORE all foreign producers, wineries, distilleries, and any company outside the US. "
                    "A valid US entry has a company name AND a US city AND a 2-letter state abbreviation "
                    "(e.g. ', NY', ', CA', ', FL', ', OR', ', TX'). "
                    "Look for key phrases: 'IMPORTED BY:', 'SOLE IMPORTER:', 'IMPORTED AND BOTTLED BY:', "
                    "'BOTTLED BY:', 'DISTRIBUTED BY:', or a US company address printed in small text. "
                    "Return the complete entry exactly as printed on the label. "
                    "If no US importer or bottler is present, reply with exactly: none"
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        },
    )
    resp.raise_for_status()
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present", "not listed"):
        return None
    # Only accept if it contains a US address tail — guards against the model
    # returning the foreign producer again or a generic non-address string
    return result if _US_ADDRESS_TAIL_RE.search(result) else None


async def _call_ollama(
    client: httpx.AsyncClient, image_b64: str, prompt: str
) -> str:
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


_FIELD_NAMES = {
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "country_of_origin", "government_warning", "contains_sulfites",
}

_SINGLE_LINE_FIELDS = {"brand_name", "class_type", "alcohol_content", "net_contents", "country_of_origin", "contains_sulfites"}
_NULL_SENTINELS = {"none", "null", "n/a", "na", "[none]", "unknown", "-"}
_INVALID_COUNTRIES = {"american", "domestic", "imported", "local"}

# Words that appear in beverage types — used to detect if class_type is actually a product name
_BEVERAGE_TYPE_WORDS = frozenset({
    "ale", "beer", "lager", "stout", "porter", "ipa",
    "whisky", "whiskey", "bourbon", "rye", "scotch",
    "wine", "champagne", "prosecco", "cider",
    "rum", "vodka", "gin", "tequila", "mezcal", "brandy",
    "mead", "liqueur", "spirits", "schnapps", "malt",
    "saison", "pilsner", "bock", "seltzer",
})

# Keyword → canonical class_type category mapping
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

# Company-type words that identify a producer entity, not a product
_COMPANY_TYPE_RE = re.compile(
    r'\b(distillery|brewery|winery|vineyard|estate)\b', re.IGNORECASE
)

# Phrases that precede the origin country on the label
_ORIGIN_PREFIX_RE = re.compile(
    r'^(?:produced?\s+in|product\s+of|made\s+in|imported?\s+from)\s+',
    re.IGNORECASE,
)

# Matches a US address tail: ", ST" or ", ST 12345" (2-letter state, optional ZIP)
# Used to detect whether an extracted producer address is domestic or foreign
_US_ADDRESS_TAIL_RE = re.compile(
    r',\s*[A-Z]\.?[A-Z]\.?(?:\s+\d{5}(?:-\d{4})?)?\s*$'
)


def _normalize_class_type(value: str) -> Optional[str]:
    """Map any class_type string to one of the three canonical categories, or None if unrecognizable."""
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

    # Strip leading origin phrases from country_of_origin to get the bare country name
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

    # Ensure government_warning includes the required prefix
    gw = data.get("government_warning")
    if gw and not gw.upper().lstrip().startswith("GOVERNMENT WARNING"):
        data["government_warning"] = "GOVERNMENT WARNING: " + gw.strip()

    # Fallback: scan other fields for sulfite mentions the model may have misattributed
    if not data.get("contains_sulfites"):
        for val in data.values():
            if isinstance(val, str) and "sulfite" in val.lower():
                for segment in re.split(r"[\n;]", val):
                    if "sulfite" in segment.lower():
                        data["contains_sulfites"] = segment.strip()
                        break
                if data.get("contains_sulfites"):
                    break

    # Null out brand_name if it's actually the producer name
    if data.get("brand_name") and data.get("producer_name_address"):
        bn_lower = data["brand_name"].strip().lower()
        prod_lower = data["producer_name_address"].strip().lower()
        if _COMPANY_TYPE_RE.search(data["brand_name"]) and bn_lower in prod_lower:
            data["brand_name"] = None

    # If brand_name is null and class_type doesn't contain any standard beverage-type
    # word, the model likely put the product name in the wrong field — swap them.
    if not data.get("brand_name") and data.get("class_type"):
        ct_lower = data["class_type"].lower()
        if not any(word in ct_lower for word in _BEVERAGE_TYPE_WORDS):
            data["brand_name"] = data["class_type"]
            data["class_type"] = None

    # Normalize class_type to one of three canonical categories; clear if unrecognizable
    ct = data.get("class_type")
    if isinstance(ct, str):
        normalized = _normalize_class_type(ct)
        data["class_type"] = normalized  # None if unrecognizable → triggers fallback call

    # Last-resort brand_name fallback for import labels
    if not data.get("brand_name") and data.get("producer_name_address"):
        producer = data["producer_name_address"]
        stripped = re.sub(
            r'^\s*(?:BOTTLED|IMPORTED|PRODUCED|DISTRIBUTED|BREWED|PACKED)\s+BY:?\s*',
            '', producer, flags=re.IGNORECASE,
        ).strip()
        name_part = re.sub(r',?\s+[\w\s]{2,},\s+[A-Z]{2}\s*$', '', stripped).strip()
        if (name_part
                and name_part.lower() != producer.strip().lower()
                and len(name_part) > 2
                and not _COMPANY_TYPE_RE.search(name_part)):
            data["brand_name"] = name_part

    return data
