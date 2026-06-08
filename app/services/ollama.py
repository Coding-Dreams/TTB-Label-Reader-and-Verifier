import asyncio
import io
import logging
import os
import re
import base64
import json
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import httpx
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

# Shared pool for OCR strategies — pytesseract calls a subprocess, so it releases
# the GIL; threading parallelises wall-time on multi-strategy cascades.
_OCR_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gw-ocr-")

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
  "us_importer": "string or null",
  "country_of_origin": "string or null",
  "government_warning": "string or null"
}

Rules:
- Return the exact text as it appears on the label for all fields except class_type
- brand_name: the trade name or trademark that identifies the brand — NOT the product series, style, or variety descriptor (e.g. the brand is "Modelo", not "Modelo Negra Especial"; "Harold's Gin", not "Single Barrel Gin"). Also do NOT capture the producer, brewery, winery, or distillery company name
- class_type: EXACTLY one of three values — "Wine", "Malt Beverage", or "Distilled Spirits". Wine = grape/fruit wines, champagne, prosecco, cider. Malt Beverage = beer, ale, lager, stout, porter, IPA, hard seltzer. Distilled Spirits = whiskey, bourbon, rum, vodka, gin, tequila, brandy, liqueur, and similar spirits. IMPORTANT: if the label shows any distilled spirit name (VODKA, GIN, RUM, WHISKEY, TEQUILA, etc.) classify as "Distilled Spirits" even if it is a flavored or canned cocktail — only use "Malt Beverage" if no distilled spirit name is present
- alcohol_content: the ABV percentage as printed (e.g. "45% ALC/VOL", "13% BY VOL")
- net_contents: the TOTAL container size (e.g. "750 ML", "100 mL", "1 PINT") — the full bottle/can volume, NOT the alcohol-per-serving amount
- contains_sulfites: search ALL panels for any sulfite statement, accepting both American ('sulfite') and British ('sulphite') spellings. This includes BOTH positive declarations (e.g. "CONTAINS SULFITES", "Contains Sulphites", "Contains Sulfating Agents") AND negative declarations (e.g. "SULFITE FREE", "NO SULFITES ADDED", "Contains No Detectable Sulphites"); return the exact text if found, null if absent
- producer_name_address: the winery, distillery, brewery, or bottler that made or bottled this product, with their address. For DOMESTIC US products this is the US producer/bottler (e.g. "BIG EASY BLENDS LLC, KENNER, LA"). For IMPORTED products put only the FOREIGN producer here (e.g. "CHATEAU DUPONT, BORDEAUX, FRANCE") — do NOT put the US importer here, use us_importer for that
- us_importer: for IMPORTED products only — the US IMPORTER, BOTTLER, or DISTRIBUTOR with a United States city and state; look for phrases like "IMPORTED BY:", "SOLE IMPORTER:", "IMPORTED AND BOTTLED BY:", "DISTRIBUTED BY:" followed by a US company name and address (e.g. "IMPORTED BY: ACME SPIRITS, MIAMI, FL"); null for domestic US products or if no US importer is listed
- country_of_origin: the country name, but ONLY if explicitly stated as the product's origin (e.g. "Product of Canada", "Made in Germany", "Imported from France"). Do NOT infer from the beverage category or style name — "American Red Wine" does NOT mean country_of_origin is "United States"
- government_warning: return exactly "GOVERNMENT WARNING" (those two words, all uppercase) if the label contains that phrase in all uppercase letters; otherwise null. Do NOT copy the warning body text and IGNORE the warning body text. ONLY focus on the GOVERNMENT WARNING.

Example output for an imported cognac label:
{
  "brand_name": "FORCE 53",
  "class_type": "Distilled Spirits",
  "alcohol_content": "53% ALC/VOL (106 PROOF)",
  "net_contents": "750 ML",
  "contains_sulfites": null,
  "producer_name_address": "H. MOUNIER, JARNAC, FRANCE",
  "us_importer": "IMPORTED BY: SIDNEY FRANK IMPORTING CO., INC., NEW ROCHELLE, NY 10801",
  "country_of_origin": "France",
  "government_warning": "GOVERNMENT WARNING"
}"""


def _auto_orient(img: Image.Image, back_panel: bool = False) -> Image.Image:
    """Correct image orientation: EXIF metadata first, pytesseract OSD for physical rotation.

    On front labels we only fix 180° flips — 90°/270° OSD calls are often mis-detections
    on graphics-heavy front art. Back panels are dominated by regulatory text, so OSD is
    much more reliable; for those we trust 90°/270° as well.
    """
    img = ImageOps.exif_transpose(img)
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0))
        if back_panel and rotate in (90, 180, 270):
            img = img.rotate(rotate, expand=True)
        elif rotate == 180:
            img = img.rotate(180, expand=True)
    except Exception:
        pass  # tesseract unavailable or insufficient text for OSD — proceed as-is
    return img


def _crop_to_content(img: Image.Image, padding: int = 20) -> Image.Image:
    """Crop away white/near-white scanner margins so content fills the resolution budget."""
    gray = img.convert("L")
    # Pixels darker than 245 are label content; 245+ is scanner white/near-white background
    content_mask = gray.point(lambda p: 255 if p < 245 else 0)
    bbox = content_mask.getbbox()
    if bbox is None:
        return img
    x0, y0, x1, y1 = bbox
    w, h = img.size
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    # Skip if less than 5% of the image would be removed (not worth the crop)
    if (x1 - x0) * (y1 - y0) > 0.95 * w * h:
        return img
    return img.crop((x0, y0, x1, y1))


def _encode_image(
    image_path: Path,
    max_side: int = _MAX_SIDE,
    enhance: bool = False,
    back_panel: bool = False,
) -> str:
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        img = _auto_orient(img, back_panel=back_panel)
        img = _crop_to_content(img)
        if enhance:
            img = ImageEnhance.Contrast(img).enhance(1.8)
            img = ImageEnhance.Sharpness(img).enhance(1.5)
        else:
            img = ImageEnhance.Contrast(img).enhance(1.15)  # light baseline for primary pass
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
        front_rgb = _auto_orient(front_img.convert("RGB"))
        back_rgb = _auto_orient(back_img.convert("RGB"), back_panel=True)

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
    image_path: Path,
    back_image_path: Optional[Path] = None,
    debug_info: Optional[dict] = None,
) -> LabelFields:
    stitched: Optional[Path] = None
    loop = asyncio.get_running_loop()
    # Kick off OCR scans immediately — they only need the raw image paths and run in
    # a thread pool, so they overlap with image stitching, the main VLM call, and
    # every secondary VLM pass. Awaited just before the final return.
    gw_ocr_future = loop.run_in_executor(
        None, _detect_government_warning_ocr, image_path, back_image_path
    )
    sulfite_present_future = loop.run_in_executor(
        None, _label_mentions_sulfite_ocr, image_path, back_image_path
    )
    try:
        back_b64: Optional[str] = None
        if back_image_path and back_image_path.exists():
            stitched = await loop.run_in_executor(
                None, _stitch_images, image_path, back_image_path
            )
            effective_path = stitched
            # Encode back panel separately at full resolution for targeted second-pass lookups
            back_b64 = await loop.run_in_executor(
                None, _encode_image, back_image_path, _MAX_SIDE, False, True
            )
        else:
            effective_path = image_path

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            image_b64 = await loop.run_in_executor(None, _encode_image, effective_path)
            raw = await _call_ollama(client, image_b64, _FULL_PROMPT)

            parsed = _parse_json(raw)
            if debug_info is not None:
                debug_info["main_raw_json"] = raw
                debug_info["after_parse"] = dict(parsed)

            data = _postprocess(parsed)
            if debug_info is not None:
                debug_info["after_postprocess"] = dict(data)
                debug_info["secondary"] = {}

            # Back panel often carries sulfite statements and regulatory text.
            # Encode at higher resolution for the dedicated sulfite scan — small-print
            # declarations are frequently missed at the default 768px.
            if back_image_path and back_image_path.exists():
                sulfite_b64 = await loop.run_in_executor(
                    None, _encode_image, back_image_path, 1024, True, True
                )
            else:
                sulfite_b64 = image_b64
            if not data.get("contains_sulfites"):
                sink: Optional[dict] = {} if debug_info is not None else None
                result = await _extract_sulfites(client, sulfite_b64, _debug_sink=sink)
                if debug_info is not None:
                    debug_info["secondary"]["sulfites"] = {"raw": sink.get("raw"), "accepted": result}
                data["contains_sulfites"] = result

            if not data.get("brand_name"):
                sink = {} if debug_info is not None else None
                result = await _extract_brand_name(client, image_b64, _debug_sink=sink)
                if debug_info is not None:
                    debug_info["secondary"]["brand_name"] = {"raw": sink.get("raw"), "accepted": result}
                data["brand_name"] = result

            # Always re-extract class_type from the front panel using the dedicated prompt.
            # The full prompt runs on the stitched image where each panel is half-width; the
            # dedicated function on the front panel alone is more reliable and applies equally
            # to every label regardless of what the full prompt returned.
            front_b64 = await loop.run_in_executor(None, _encode_image, image_path, 1024)
            sink = {} if debug_info is not None else None
            class_from_front = await _extract_class_type(client, front_b64, _debug_sink=sink)
            if debug_info is not None:
                debug_info["secondary"]["class_type"] = {"raw": sink.get("raw"), "accepted": class_from_front}
            if class_from_front:
                data["class_type"] = class_from_front

            # If net_contents not found in main pass, try a targeted second-pass lookup
            if not data.get("net_contents"):
                net_b64 = back_b64 or image_b64
                sink = {} if debug_info is not None else None
                result = await _extract_net_contents(client, net_b64, _debug_sink=sink)
                if debug_info is not None:
                    debug_info["secondary"]["net_contents"] = {"raw": sink.get("raw"), "accepted": result}
                data["net_contents"] = result

            # If producer/bottler still missing, look for it specifically on the back panel
            if not data.get("producer_name_address"):
                # Use the high-resolution + contrast-enhanced back panel encoding so
                # small-print producer/bottler text (e.g. COLA17/18 'DC FLYNT MW SELECTIONS')
                # remains legible to the VLM. Falls back to stitched image when no back exists.
                prod_b64 = sulfite_b64 if back_image_path and back_image_path.exists() else image_b64
                sink = {} if debug_info is not None else None
                result = await _extract_producer(client, prod_b64, _debug_sink=sink)
                if debug_info is not None:
                    debug_info["secondary"]["producer_name_address"] = {"raw": sink.get("raw"), "accepted": result}
                if result:
                    result = _IMPORTER_PREFIX_RE.sub('', result).strip()
                    result = _URL_SUFFIX_RE.sub('', result).strip()
                    data["producer_name_address"] = result or None

            # If producer is a non-US address, the US importer was missed in the main pass —
            # run a dedicated importer lookup on the full stitched image
            if data.get("producer_name_address") and not _is_us_address(data["producer_name_address"]):
                sink = {} if debug_info is not None else None
                importer = await _extract_importer(client, image_b64)
                if debug_info is not None:
                    debug_info["secondary"]["us_importer"] = {"raw": importer, "accepted": importer}
                if importer:
                    cleaned = _IMPORTER_PREFIX_RE.sub('', importer).strip()
                    cleaned = _URL_SUFFIX_RE.sub('', cleaned).strip()
                    data["producer_name_address"] = cleaned or importer

            # OCR-based government_warning: exact uppercase check overrides VLM output.
            # Prevents both false positives (model normalises lowercase text to uppercase)
            # and false negatives (model misses small-print text the OCR can still read).
            # Dispatched after _postprocess and overlapped with the VLM secondary passes.
            gw_ocr = await gw_ocr_future
            data["government_warning"] = gw_ocr
            if debug_info is not None:
                debug_info["secondary"]["government_warning"] = {"raw": gw_ocr, "accepted": gw_ocr}

            # OCR-based sulfite gate: if the VLM said the label contains sulfites but
            # OCR finds no 'sulfite'/'sulphite' anywhere, the VLM hallucinated and we
            # null the field. We don't override when the VLM said None (Phase B
            # compliance test will flag wines missing the declaration anyway).
            sulfite_present = await sulfite_present_future
            if data.get("contains_sulfites") and not sulfite_present:
                if debug_info is not None:
                    debug_info["secondary"]["contains_sulfites_ocr_gate"] = {
                        "vlm_value": data["contains_sulfites"],
                        "ocr_found_mention": False,
                        "result": None,
                    }
                data["contains_sulfites"] = None

            return LabelFields(**data)
    finally:
        if stitched:
            stitched.unlink(missing_ok=True)


async def _extract_brand_name(
    client: httpx.AsyncClient, image_b64: str, _debug_sink: Optional[dict] = None
) -> Optional[str]:
    resp = await _post_with_retry(client, {
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
        })
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if _debug_sink is not None:
        _debug_sink["raw"] = result
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present", "cannot determine"):
        return None
    return result or None


async def _extract_class_type(
    client: httpx.AsyncClient, image_b64: str, _debug_sink: Optional[dict] = None
) -> Optional[str]:
    resp = await _post_with_retry(client, {
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
        })
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if _debug_sink is not None:
        _debug_sink["raw"] = result
    if result.lower() in _NULL_SENTINELS:
        return None
    return _normalize_class_type(result) or result or None


async def _extract_sulfites(
    client: httpx.AsyncClient, image_b64: str, _debug_sink: Optional[dict] = None
) -> Optional[str]:
    resp = await _post_with_retry(client, {
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    'Look at this alcohol label image carefully. '
                    'Search every panel for any sulfite statement, accepting BOTH '
                    'American ("sulfite") and British ("sulphite") spellings. '
                    'This includes positive statements like "CONTAINS SULFITES", '
                    '"Contains Sulphites", "Contains Sulfating Agents", AND negative '
                    'statements like "SULFITE FREE", "NO SULPHITES ADDED", '
                    '"Contains No Detectable Sulphites". '
                    'Reply with just that exact text if you find it, or reply with '
                    'the single word "none" if no sulfite statement is present.'
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        })
    result = resp.json()["message"]["content"].strip()
    if _debug_sink is not None:
        _debug_sink["raw"] = result
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present", "absent"):
        return None
    return result


async def _extract_importer(client: httpx.AsyncClient, image_b64: str) -> Optional[str]:
    resp = await _post_with_retry(client, {
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
        })
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    is_null = result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present", "not listed")
    if is_null:
        logger.warning("_extract_importer raw=%r accepted=None (null sentinel)", result)
        return None
    # Accept if it has a US address tail (city, STATE) OR a US corporate identifier
    # (LLC, Inc., L.L.C., etc.) — some importers are printed without a full address.
    has_us_address = bool(_US_ADDRESS_TAIL_RE.search(result))
    has_us_corp = bool(_US_COMPANY_RE.search(result))
    accepted = has_us_address or has_us_corp
    logger.warning("_extract_importer raw=%r us_address=%s us_corp=%s accepted=%s", result, has_us_address, has_us_corp, accepted)
    return result if accepted else None


async def _extract_producer(
    client: httpx.AsyncClient, image_b64: str, _debug_sink: Optional[dict] = None
) -> Optional[str]:
    resp = await _post_with_retry(client, {
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    "Look at this alcohol beverage label. "
                    "Find the PRODUCER, BOTTLER, BREWER, or MANUFACTURER — the company "
                    "that made or packaged this product — with their full address. "
                    "Look for phrases like 'PRODUCED BY:', 'BOTTLED BY:', 'BREWED BY:', "
                    "'PRODUCED & PACKAGED BY:', 'MANUFACTURED BY:', 'BREWED AND BOTTLED BY:', "
                    "or a company name followed by a city and state/country. "
                    "Return the complete entry exactly as printed on the label, "
                    "including any prefix such as 'PRODUCED BY:'. "
                    "If not found, reply with exactly: none"
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        })
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if _debug_sink is not None:
        _debug_sink["raw"] = result
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present"):
        return None
    return result or None


async def _extract_net_contents(
    client: httpx.AsyncClient, image_b64: str, _debug_sink: Optional[dict] = None
) -> Optional[str]:
    resp = await _post_with_retry(client, {
            "model": MODEL,
            "messages": [{"role": "user",
                "content": (
                    "Look at this alcohol beverage label. "
                    "Find the NET CONTENTS — the TOTAL volume of liquid in the container "
                    "(e.g. '100 mL', '750 mL', '1.75 L'). "
                    "Look near the barcode, bottom edge, or nutrition facts panel. "
                    "Return only the volume with units. "
                    "If not found, reply with exactly: none"
                ),
                "images": [image_b64]}],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        })
    result = resp.json()["message"]["content"].strip().split("\n")[0].strip()
    if _debug_sink is not None:
        _debug_sink["raw"] = result
    if result.lower() in _NULL_SENTINELS or result.lower() in ("no", "not found", "not present"):
        return None
    if not re.search(r'\d+\.?\d*\s*(?:ml|l\b|fl\.?\s*oz)', result, re.IGNORECASE):
        return None
    return result


_TRANSIENT_HTTPX = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.PoolTimeout,
)


async def _post_with_retry(
    client: httpx.AsyncClient, json_body: dict, attempts: int = 3, backoff: float = 5.0
) -> httpx.Response:
    """POST to Ollama /api/chat with retry on transient connection / read errors.

    The Ollama container occasionally drops connections mid-request under load
    ('Server disconnected without sending a response'); retrying lets the call
    survive a single bad attempt rather than failing the whole verification.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=json_body)
            resp.raise_for_status()
            return resp
        except _TRANSIENT_HTTPX as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            logger.warning(
                "Ollama transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, attempts, backoff, exc,
            )
            await asyncio.sleep(backoff)
    raise last_exc  # unreachable, but satisfies type checker


async def _call_ollama(
    client: httpx.AsyncClient, image_b64: str, prompt: str
) -> str:
    resp = await _post_with_retry(
        client,
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "format": "json",
            "keep_alive": -1,
            "options": {"temperature": 0.1},
        },
        attempts=3,
        backoff=15.0,
    )
    return resp.json()["message"]["content"]


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


_FIELD_NAMES = {
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "producer_name_address", "us_importer", "country_of_origin", "government_warning", "contains_sulfites",
}

_SINGLE_LINE_FIELDS = {"brand_name", "class_type", "alcohol_content", "net_contents", "country_of_origin", "contains_sulfites", "us_importer"}
_NULL_SENTINELS = {"none", "null", "n/a", "na", "[none]", "unknown", "-"}
_INVALID_COUNTRIES = {"american", "domestic", "imported", "local"}

# country_of_origin is meant to mark FOREIGN origin only. Anything resolving to the
# United States — country name variants, individual state names, state abbreviations —
# is nulled so domestic labels show country_of_origin as None (matching truth files).
_US_LOCATIONS = frozenset({
    # Country variants
    "united states", "united states of america", "america", "usa", "u.s.", "u.s.a.",
    "us", "the usa", "the united states", "the u.s.", "the u.s.a.",
    # 2-letter state abbreviations
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il",
    "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt",
    "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri",
    "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc",
    # Full state names
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", "illinois",
    "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana",
    "nebraska", "nevada", "new hampshire", "new jersey", "new mexico", "new york",
    "new york state", "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "washington state",
    "west virginia", "wisconsin", "wyoming", "district of columbia",
})

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

# Matches a US address tail: ", STATE" or ", STATE ZIPCODE"
# Accepts both 2-letter abbreviations (NY, CA) and full state names (New York, California).
# Full names prevent false negatives on labels that spell out the state.
# Explicit name list prevents false positives on European 5-digit postal codes.
_US_ADDRESS_TAIL_RE = re.compile(
    r',\s*(?:'
    r'AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|'
    r'NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC|'
    r'Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|'
    r'Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|'
    r'Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|'
    r'Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|'
    r'North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|'
    r'Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|'
    r'Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming|'
    r'District\s+of\s+Columbia'
    r')(?:\s+\d{5}(?:-\d{4})?)?\s*$',
    re.IGNORECASE
)

# Strips a trailing country/country-code suffix before address matching
_TRAILING_USA_RE = re.compile(r',?\s*U\.?S\.?A?\.?\s*$', re.IGNORECASE)
# US corporate entity suffixes — accept importer results that name a US company
# even when the label omits the city/state address
_US_COMPANY_RE = re.compile(r'\b(?:LLC|L\.L\.C\.|Inc\.?|Corp\.?|Ltd\.?|Co\.)\b', re.IGNORECASE)
# Strips "IMPORTED BY:", "IMPORTED EXCLUSIVELY BY:", "BOTTLED BY:",
# "PRODUCED & PACKAGED BY:", etc. prefixes from producer/importer values.
_IMPORTER_PREFIX_RE = re.compile(
    r'^\s*(?:IMPORTED|BOTTLED|DISTRIBUTED|PRODUCED|PACKED|MADE)'
    r'(?:\s+(?:AND|&)\s+\w+)?(?:\s+EXCLUSIVELY)?'
    r'\s+BY:?\s*',
    re.IGNORECASE,
)
# Strips trailing website URLs (e.g. " www.ourniche.com")
_URL_SUFFIX_RE = re.compile(r'\s+(?:www|http)\.\S+.*$', re.IGNORECASE)
# Normalizes dotted state abbreviations like N.Y. or D.C. to NY / DC
_DOTTED_ABBREV_RE = re.compile(r'\b([A-Z])\.([A-Z])\.?\s*$')
# Detects "GOVERNMENT WARNING" with possible line-break between the two words
_GOVT_WARNING_RE = re.compile(r'GOVERNMENT\s+WARNING')
# Matches both 'sulfite' (American) and 'sulphite' (British). The 'h' after 'p'
# is required only in the British form; the 'f' form has no h.
_SULFITE_MENTION_RE = re.compile(r'\bsul(?:f|ph)ite', re.IGNORECASE)


def _ocr_finds_warning(img, config: str = "") -> bool:
    """Single OCR call — True if exact uppercase 'GOVERNMENT WARNING' appears in result."""
    import pytesseract
    text = pytesseract.image_to_string(img, config=config)
    return bool(_GOVT_WARNING_RE.search(text))


def _label_mentions_sulfite_ocr(image_path: Path, back_image_path: Optional[Path]) -> bool:
    """Quick OCR scan — True if any 'sulfite' or 'sulphite' token appears anywhere.

    Cheaper than the gov_warning cascade (case-insensitive substring instead of exact
    phrase, fewer fallback strategies) because false positives here are harmless: this
    is a *gate* to suppress VLM sulfite hallucinations on labels that don't mention
    sulfites at all (e.g. COLA17/18 KIRKLAND lime drop, where 'CONTAINS ALCOHOL' tricks
    the VLM into outputting 'CONTAINS SULFITES').
    """
    try:
        import pytesseract
        for path in filter(None, [image_path, back_image_path]):
            if not path.exists():
                continue
            img = ImageOps.exif_transpose(Image.open(path)).convert("L")
            for variant in (img, img.resize((img.width * 2, img.height * 2), Image.LANCZOS)):
                if _SULFITE_MENTION_RE.search(pytesseract.image_to_string(variant)):
                    return True
        return False
    except Exception:
        return True  # OCR unavailable — don't override the VLM result


def _any_match_parallel(tasks: list) -> bool:
    """Submit (img, config) OCR tasks to the shared pool. True if any finds the warning.

    Each future's exception is handled independently — one crashed tesseract subprocess
    must not poison the result of the other strategies. Otherwise a transient subprocess
    failure under load makes the whole detection silently return None.
    """
    futures = [_OCR_POOL.submit(_ocr_finds_warning, img, cfg) for img, cfg in tasks]
    for f in as_completed(futures):
        try:
            if f.result():
                for pending in futures:
                    if pending is not f and not pending.done():
                        pending.cancel()
                return True
        except Exception as exc:
            logger.warning("OCR strategy failed (continuing with others): %s", exc)
    return False


def _detect_government_warning_ocr(image_path: Path, back_image_path: Optional[Path]) -> Optional[str]:
    """OCR-based presence check: returns 'GOVERNMENT WARNING' only if exact uppercase phrase found.

    Each tier runs its strategies in parallel via a shared thread pool, falling through to
    the next tier only if no strategy in the current one found the phrase. Most labels
    resolve in tier 1 in well under a second.
    """
    try:
        import pytesseract  # noqa: F401  (fail fast if missing)
        prepared = []
        for path in filter(None, [image_path, back_image_path]):
            if not path.exists():
                continue
            rgb = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            gray = rgb.convert("L")
            up2 = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
            prepared.append({"rgb": rgb, "gray": gray, "up2": up2})
        if not prepared:
            return None

        # Tier 1: cheap whole-image OCR on each source
        tier1 = []
        for p in prepared:
            tier1.extend([
                (p["gray"], ""),
                (p["up2"], ""),
                (p["up2"], "--psm 6"),
                (p["gray"], "--psm 12"),
            ])
        if _any_match_parallel(tier1):
            return "GOVERNMENT WARNING"

        # Tier 2: rotated whole image (180° upside-down, 90°/270° landscape labels)
        tier2 = []
        for p in prepared:
            for angle in (180, 270, 90):
                rot = p["gray"].rotate(angle, expand=True)
                tier2.append((rot.resize((rot.width * 2, rot.height * 2), Image.LANCZOS), ""))
        if _any_match_parallel(tier2):
            return "GOVERNMENT WARNING"

        # Tier 3: single-colour channels — handles green-on-green, red-on-red labels
        tier3 = []
        for p in prepared:
            for ch in p["rgb"].split():
                ch_up = ch.resize((ch.width * 2, ch.height * 2), Image.LANCZOS)
                tier3.append((ch_up, "--psm 6"))
                rot = ch.rotate(180, expand=True)
                tier3.append((rot.resize((rot.width * 2, rot.height * 2), Image.LANCZOS), "--psm 6"))
        if _any_match_parallel(tier3):
            return "GOVERNMENT WARNING"

        # Tier 4: right-edge crop + perpendicular rotation (TOMMYROTTER-style sideways text)
        tier4 = []
        for p in prepared:
            w, h = p["rgb"].size
            for left_pct, top_pct in ((0.85, 0.30), (0.80, 0.0)):
                region = p["rgb"].crop((int(w * left_pct), int(h * top_pct), w, h))
                rr, gg, _ = region.split()
                for ch in (gg, rr):
                    rot = ch.rotate(-90, expand=True)
                    scaled = rot.resize((rot.width * 4, rot.height * 4), Image.LANCZOS)
                    tier4.append((scaled, "--psm 6"))
        if _any_match_parallel(tier4):
            return "GOVERNMENT WARNING"

        return None
    except Exception:
        return None


def _is_us_address(addr: str) -> bool:
    """Return True if addr ends with a recognisable US location."""
    if _US_ADDRESS_TAIL_RE.search(addr):
        return True
    # Handle dotted abbreviations: N.Y. → NY, D.C. → DC
    norm = _DOTTED_ABBREV_RE.sub(r'\1\2', addr.strip())
    if _US_ADDRESS_TAIL_RE.search(norm):
        return True
    # Handle trailing country suffix: "Tennessee, USA" → "Tennessee"
    stripped = _TRAILING_USA_RE.sub('', addr.strip())
    return bool(_US_ADDRESS_TAIL_RE.search(stripped))


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

    # Null out values that are JSON key names (model confused structure with content).
    # Compare without space→underscore conversion: "CONTAINS SULFITES" must not match
    # the field name "contains_sulfites" and be erroneously zeroed out.
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, str) and val.strip().lower() in _FIELD_NAMES:
            data[key] = None

    # Strip leading origin phrases from country_of_origin to get the bare country name
    country = data.get("country_of_origin")
    if isinstance(country, str):
        stripped = _ORIGIN_PREFIX_RE.sub("", country.strip()).strip()
        if stripped != country.strip():
            data["country_of_origin"] = stripped
            country = stripped

    # Null out country_of_origin if it's not an actual foreign country name.
    # country_of_origin is for IMPORTED products only — domestic US labels (whether
    # the model said "United States", "USA", or a state name) should resolve to None.
    country = data.get("country_of_origin")
    if isinstance(country, str):
        normalized = country.strip().lower()
        if (
            re.fullmatch(r"[A-Z]{2}", country.strip())
            or normalized in _INVALID_COUNTRIES
            or normalized in _US_LOCATIONS
        ):
            data["country_of_origin"] = None

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

    # If the model extracted a dedicated US importer, it takes precedence over
    # the foreign producer — strip the "IMPORTED BY:" prefix and any trailing URL,
    # then merge into the single producer_name_address field.
    _pre_swap_producer = (data.get("producer_name_address") or "").strip().lower()
    us_importer = data.pop("us_importer", None)
    if us_importer:
        cleaned = _IMPORTER_PREFIX_RE.sub('', us_importer).strip()
        cleaned = _URL_SUFFIX_RE.sub('', cleaned).strip()
        data["producer_name_address"] = cleaned or us_importer

        # For imported labels: if brand_name starts the pre-swap producer string
        # (e.g. "Felline" in "FELLINE Soc. Agr. a r. l., MANDURIA, ITALIA"), the
        # model captured the bottler entity instead of the product brand — null it
        # so the secondary brand_name pass finds the actual label text.
        bn = (data.get("brand_name") or "").strip().lower()
        n = len(bn)
        if bn and _pre_swap_producer and _pre_swap_producer.startswith(bn) and (
            n == len(_pre_swap_producer) or not _pre_swap_producer[n].isalnum()
        ):
            data["brand_name"] = None

    # Last-resort brand_name fallback: extract the company name from a foreign producer
    # address when no brand was found elsewhere. Skip US addresses — after the importer
    # swap the producer field holds the US importer company, not the product brand.
    if not data.get("brand_name") and data.get("producer_name_address"):
        producer = data["producer_name_address"]
        if not _is_us_address(producer):
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
