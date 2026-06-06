import os
import re
import base64
import json
import httpx
from pathlib import Path
from typing import Optional

from app.models.label import LabelFields

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "glm-ocr"
TIMEOUT = 120.0

_EXTRACTION_PROMPT = """You are an OCR assistant specialized in reading alcohol beverage labels.
Extract the following fields from this label image and return ONLY valid JSON with no additional text.

Required JSON format:
{
  "brand_name": "string or null",
  "class_type": "string or null",
  "alcohol_content": "string or null",
  "net_contents": "string or null",
  "producer_name_address": "string or null",
  "country_of_origin": "string or null",
  "government_warning": "string or null"
}

Rules:
- Copy the exact text as it appears on the label for each field
- Set any field to null if not visible or not present on the label
- Return ONLY the JSON object, no explanation or markdown

Field guidance:
- brand_name: the product or distillery name (e.g., "Jack Daniel's", "ABC Distillery")
- class_type: the beverage category printed on the label (e.g., "Straight Rye Whisky", "American Red Wine", "Rum with Coconut Liqueur") — NEVER a JSON key name
- alcohol_content: the ABV as printed (e.g., "45% ALC/VOL", "13% BY VOL")
- net_contents: the volume as printed (e.g., "750 ML", "1 PINT", "200 ML")
- producer_name_address: distiller, bottler, or importer name and address as printed
- country_of_origin: country where produced or imported from
- government_warning: copy the COMPLETE warning text EXACTLY as printed, including the "GOVERNMENT WARNING:" heading — this heading MUST be included if it appears on the label

CRITICAL: Values must be text read from the label image. Never use JSON key names (brand_name, class_type, alcohol_content, etc.) as values."""

_STRICT_PROMPT = _EXTRACTION_PROMPT + "\n\nCRITICAL: Your response must begin with { and end with }. No other characters outside the JSON."


async def extract_label_fields(image_path: Path) -> LabelFields:
    image_data = base64.b64encode(image_path.read_bytes()).decode()
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        raw = await _call_ollama(client, image_data, _EXTRACTION_PROMPT)
        try:
            return LabelFields(**_parse_json(raw))
        except (json.JSONDecodeError, ValueError):
            raw = await _call_ollama(client, image_data, _STRICT_PROMPT)
            return LabelFields(**_parse_json(raw))


async def _call_ollama(client: httpx.AsyncClient, image_data: str, prompt: str) -> str:
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
            "format": "json",
        },
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # Extract JSON object if wrapped in markdown fences
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return _postprocess(json.loads(raw))


_FIELD_NAMES = {
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "country_of_origin", "government_warning",
}

def _postprocess(data: dict) -> dict:
    # Normalize empty strings to None
    for key in list(data.keys()):
        if isinstance(data[key], str) and not data[key].strip():
            data[key] = None

    # If any field's value is a JSON key name, the model confused structure with content
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, str) and val.strip().lower().replace(" ", "_") in _FIELD_NAMES:
            data[key] = None

    # Ensure government_warning includes the required "GOVERNMENT WARNING:" prefix
    gw = data.get("government_warning")
    if gw and not gw.upper().lstrip().startswith("GOVERNMENT WARNING"):
        data["government_warning"] = "GOVERNMENT WARNING: " + gw.strip()

    return data
