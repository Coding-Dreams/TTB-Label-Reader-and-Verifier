#!/usr/bin/env python3
"""
Benchmark glm-ocr vs pytesseract for GOVERNMENT WARNING detection.

Usage:
  1. From the host, pull the model into the running Ollama container:
       docker compose exec ollama ollama pull glm-ocr
  2. Expose Ollama port temporarily (edit docker-compose.yml: add "11434:11434"
     to the ollama service's ports), then `docker compose up -d ollama` --
     or run this script from inside the api container:
       docker compose exec api python /app/scripts/test_glm_ocr.py
  3. Compare wall-clock and accuracy against the pytesseract cascade.
"""
import base64
import os
import re
import sys
import time
from pathlib import Path

import httpx

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = "glm-ocr"
GW_RE = re.compile(r"GOVERNMENT\s+WARNING")
ROOT = Path(__file__).parent.parent / "testLabels"

# Labels grouped by difficulty (based on pytesseract cascade tier needed)
EASY = ["COLA1", "COLA12", "COLA15", "COLA25", "COLA26", "COLA28", "COLA9", "TEST1", "TEST3"]
HARD = ["COLA8", "COLA10", "COLA11", "COLA21", "COLA6", "COLA20"]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def query_glm_ocr(image_b64: str) -> str:
    """Ask glm-ocr to transcribe the entire image."""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": "Extract all text from this image, preserving case and layout.",
                "images": [image_b64],
            }],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0.0},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def detect_warning(label: str) -> tuple[bool, float, str]:
    folder = ROOT / label
    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        return (False, 0.0, "no images")
    t0 = time.time()
    snippets = []
    for img in images:
        text = query_glm_ocr(encode_image(img))
        snippets.append(text[:200])
        if GW_RE.search(text):
            return (True, time.time() - t0, f"found in {img.name}")
    return (False, time.time() - t0, f"none: {snippets!r}"[:150])


def main():
    targets = sys.argv[1:] or EASY + HARD
    print(f"Testing {len(targets)} labels against {MODEL} at {OLLAMA_URL}\n")
    total_t = 0.0
    found = 0
    for label in targets:
        try:
            ok, dur, note = detect_warning(label)
        except Exception as e:
            ok, dur, note = False, 0.0, f"ERROR: {e}"
        total_t += dur
        if ok:
            found += 1
        marker = "✓" if ok else "✗"
        print(f"  {marker} {label:<8} {dur:>5.1f}s  {note}")
    print(f"\n{found}/{len(targets)} found in {total_t:.1f}s "
          f"(avg {total_t/max(len(targets),1):.1f}s/label)")


if __name__ == "__main__":
    main()
