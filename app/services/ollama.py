import os
import re
import base64
import json
import httpx
from pathlib import Path
from typing import Optional

from app.models.label import LabelFields

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "llava-phi3"
TIMEOUT = 30.0

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
- Return the exact text as it appears on the label
- Set any field to null if not visible or not present on the label
- For government_warning, include the complete warning text
- Return ONLY the JSON object, no explanation or markdown"""

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
    return json.loads(raw)
