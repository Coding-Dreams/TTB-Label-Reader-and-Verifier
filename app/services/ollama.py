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
from pathlib import Path
from PIL import Image

from app.models.label import LabelFields

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "qwen2.5vl:3b"
TIMEOUT = 120.0

_MAX_SIDE = 768  # cap large uploads before encoding

logger = logging.getLogger(__name__)

_FULL_PROMPT = """You are an OCR assistant specialized in reading alcohol beverage labels.
Return ONLY valid JSON with no additional text. Set any field to null if not visible.

Required JSON format:
{
  "brand_name": "string or null",
  "class_type": "string or null",
  "alcohol_content": "string or null",
  "net_contents": "string or null",
  "contains_sulfites": "string or null",
  "producer_name_address": "string or null",
  "country_of_origin": "string or null",
  "government_warning": "string or null"
}

Rules:
- Return the exact text as it appears on the label
- brand_name: the consumer-facing product name as printed (e.g. "ABC Single Barrel", "Honey Huckleberry Pie") — this is the specific product or label name consumers use to identify the beverage; do NOT capture the producer, brewery, winery, or distillery company name
- class_type: the beverage category as printed (e.g. "Straight Rye Whisky", "American Red Wine", "Rum with Coconut Liqueur") — NOT the brewery or winery name
- alcohol_content: the ABV percentage as printed (e.g. "45% ALC/VOL", "13% BY VOL")
- net_contents: the volume as printed (e.g. "750 ML", "1 PINT")
- contains_sulfites: search ALL panels for any sulfite statement (e.g. "CONTAINS SULFITES", "Contains Sulfating Agents"); return the exact text if found, null if absent
- producer_name_address: the COMPLETE producer/bottler/importer entry as printed — capture BOTH the company name AND the full location (city, state/country) as one value (e.g. "ABC DISTILLERY FREDERICK, MD", "IMPORTED BY: 12345 IMPORTS MIAMI, FL"); do NOT return just the name or just the address alone
- country_of_origin: the country name, but ONLY if explicitly stated as the product's origin (e.g. "Product of Canada", "Made in Germany", "Imported from France"). Do NOT infer from the beverage category or style name — "American Red Wine" does NOT mean country_of_origin is "United States"
- government_warning: the COMPLETE warning text EXACTLY as printed, including the "GOVERNMENT WARNING:" heading if present"""


def _encode_image(image_path: Path) -> str:
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / max(w, h)
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


async def extract_label_fields(image_path: Path) -> LabelFields:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        loop = asyncio.get_event_loop()
        image_b64 = await loop.run_in_executor(None, _encode_image, image_path)
        raw = await _call_ollama(client, image_b64, _FULL_PROMPT)
        data = _postprocess(_parse_json(raw))
        if not data.get("contains_sulfites"):
            data["contains_sulfites"] = await _extract_sulfites(client, image_b64)
        return LabelFields(**data)


async def _extract_sulfites(client: httpx.AsyncClient, image_b64: str) -> Optional[str]:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    'Look at this alcohol label image carefully. '
                    'Search every panel for any text mentioning sulfites, such as '
                    '"CONTAINS SULFITES", "Contains Sulfating Agents", or similar. '
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

    # Null out country_of_origin if it's not an actual country name
    country = data.get("country_of_origin")
    if isinstance(country, str):
        if re.fullmatch(r"[A-Z]{2}", country.strip()):
            data["country_of_origin"] = None
        elif country.strip().lower() in _INVALID_COUNTRIES:
            data["country_of_origin"] = None
        else:
            # Reject if country appears to be inferred from the class/type descriptor
            # (e.g. model returning "United States" because label says "American Red Wine")
            class_type = (data.get("class_type") or "").lower()
            if country.strip().lower() in ("united states", "america") and (
                "american" in class_type or "domestic" in class_type
            ):
                data["country_of_origin"] = None

    # Ensure government_warning includes the required prefix
    gw = data.get("government_warning")
    if gw and not gw.upper().lstrip().startswith("GOVERNMENT WARNING"):
        data["government_warning"] = "GOVERNMENT WARNING: " + gw.strip()

    # Fallback: if model didn't populate contains_sulfites, scan other fields for
    # sulfite mentions (model may misattribute the statement to a nearby field)
    if not data.get("contains_sulfites"):
        for val in data.values():
            if isinstance(val, str) and "sulfite" in val.lower():
                for segment in re.split(r"[\n;]", val):
                    if "sulfite" in segment.lower():
                        data["contains_sulfites"] = segment.strip()
                        break
                if data.get("contains_sulfites"):
                    break

    return data
