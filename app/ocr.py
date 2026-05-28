"""OCR cost ladder.

Cheapest first, escalate only on failure:
  1. native PDF text  — free (PyMuPDF)
  2. Google Vision     — ~0.15c/page, capped at MAX_VISION_PAGES
  3. (caller) gpt-4o-mini vision — last resort, handled in main via llm.py

Each step reports the tier + estimated cost so the backend can track spend.
"""
from typing import List, Optional

import fitz  # PyMuPDF

from .config import get_settings


class OcrResult:
    def __init__(self, text: str, tier: str, cost: float, pages: int = 0):
        self.text = text
        self.tier = tier
        self.cost = cost
        self.pages = pages


def _vision_client():
    s = get_settings()
    try:
        from google.cloud import vision

        if s.google_vision_credentials_json:
            import json

            from google.oauth2 import service_account

            info = json.loads(s.google_vision_credentials_json)
            creds = service_account.Credentials.from_service_account_info(info)
            return vision.ImageAnnotatorClient(credentials=creds)
        # else rely on GOOGLE_APPLICATION_CREDENTIALS / ADC
        return vision.ImageAnnotatorClient()
    except Exception:
        return None


def native_pdf_text(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return ""
    try:
        return "\n".join(page.get_text() for page in doc).strip()
    finally:
        doc.close()


def render_pdf_pages_png(data: bytes, max_pages: int) -> List[bytes]:
    imgs: List[bytes] = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return imgs
    try:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(dpi=150)
            imgs.append(pix.tobytes("png"))
    finally:
        doc.close()
    return imgs


def vision_ocr_images(images: List[bytes]) -> Optional[str]:
    client = _vision_client()
    if client is None or not images:
        return None
    try:
        from google.cloud import vision

        texts = []
        for img in images:
            resp = client.document_text_detection(image=vision.Image(content=img))
            if resp.error.message:
                return None
            texts.append(resp.full_text_annotation.text or "")
        return "\n".join(texts).strip()
    except Exception:
        return None


def extract_text(data: bytes, mime: str) -> OcrResult:
    """Run the ladder for a TEXT document and return the best text we got."""
    s = get_settings()
    is_pdf = "pdf" in (mime or "").lower() or data[:5] == b"%PDF-"

    if is_pdf:
        txt = native_pdf_text(data)
        if len(txt) >= s.native_text_min_chars:
            return OcrResult(txt, "native_pdf", 0.0)
        images = render_pdf_pages_png(data, s.max_vision_pages)
        vtxt = vision_ocr_images(images)
        if vtxt:
            return OcrResult(
                vtxt, "google_vision", len(images) * s.vision_cost_cents_per_page, len(images)
            )
        return OcrResult("", "needs_vision_fallback", 0.0)

    # image input
    vtxt = vision_ocr_images([data])
    if vtxt:
        return OcrResult(vtxt, "google_vision", s.vision_cost_cents_per_page, 1)
    return OcrResult("", "needs_vision_fallback", 0.0)
