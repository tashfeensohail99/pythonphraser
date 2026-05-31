"""Split & categorize a multi-document upload into constituent documents.

OCR-first (free native PDF text, else Google Vision) + DETERMINISTIC keyword
classification over the canonical DOC_TYPES vocab (the same set
/validate-document and the backend's checklist docTypes use). No OpenAI vision
here — a low-confidence page is flagged needs_review for a human, never
auto-classified by an expensive/overconfident model.

Strategy (mirrors the proven recruitment splitter, adapted to immigration):
  - PDF -> per page: native text (free) or Google Vision OCR -> classify.
  - Group consecutive same-type pages into one logical document
    (a 6-page bank statement => one doc, pages [2,3,4,5,6,7]).
  - Confidence below CONFIDENCE_GATE -> doc_type kept but needs_review=True.
The result feeds the backend, which auto-files high-confidence segments into
their checklist slots and sends needs_review ones to the associate's reviewer.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF

from . import ocr
from .config import get_settings

# How many pages of a bundle we'll scan. Google Vision is ~0.15c/page, so 30
# pages is ~4.5c worst case — fine. Bundles longer than this get truncated
# (logged in the response) rather than silently dropped.
SPLIT_MAX_PAGES = 30
# Below this a page is split out but flagged for human review, not trusted.
CONFIDENCE_GATE = 0.62
# A page with at least this much native text is classified for free (no OCR).
NATIVE_MIN_CHARS = 40

_CNIC_RE = re.compile(r"\b\d{5}\s*[-–]?\s*\d{7}\s*[-–]?\s*\d\b")
_MRZ_RE = re.compile(r"[A-Z0-9<]{24,}")
_PK_PASSPORT_NO_RE = re.compile(r"\b[A-Z]{2}\s*\d{7}\b")

# (doc_type, [(keyword, weight), ...]). Highest total wins. Keep keywords
# lowercase; text is lowercased before matching. Regex signals handled
# separately in _classify for passport/CNIC.
_RULES: Dict[str, List[Tuple[str, int]]] = {
    "BANK_STATEMENT": [
        ("statement of account", 4), ("account statement", 4), ("closing balance", 3),
        ("opening balance", 3), ("available balance", 2), ("account number", 2),
        ("account title", 2), ("transaction", 1), ("debit", 1), ("credit", 1),
        ("branch code", 1), ("iban", 2), ("bank", 1),
    ],
    "LANGUAGE_TEST": [
        ("test report form", 5), ("ielts", 4), ("celpip", 4), ("toefl", 4),
        ("pte academic", 4), ("duolingo", 4), ("overall band", 4),
        ("listening", 1), ("reading", 1), ("writing", 1), ("speaking", 1), ("cefr", 2),
    ],
    "ACADEMIC_TRANSCRIPT": [
        ("transcript", 4), ("marksheet", 4), ("mark sheet", 4), ("statement of marks", 4),
        ("semester", 2), ("cgpa", 3), ("gpa", 2), ("credit hours", 2), ("grade point", 2),
    ],
    "EDUCATION_CERTIFICATE": [
        ("degree", 3), ("has been awarded", 4), ("diploma", 3), ("bachelor", 2),
        ("master", 2), ("convocation", 3), ("intermediate certificate", 3),
        ("secondary school certificate", 3), ("matriculation", 3), ("graduated", 2),
    ],
    "ACCEPTANCE_LETTER": [
        ("letter of acceptance", 5), ("offer of admission", 5), ("admission letter", 4),
        ("pleased to offer you admission", 5), ("enrolment", 2), ("conditional offer", 3),
    ],
    "RESUME": [
        ("curriculum vitae", 5), ("resume", 4), ("work experience", 2), ("career objective", 3),
        ("professional summary", 3), ("key skills", 2), ("references available", 2),
    ],
    "POLICE_CLEARANCE": [
        ("police clearance certificate", 5), ("character certificate", 5),
        ("no criminal record", 4), ("antecedents", 3), ("police character", 5),
        ("superintendent of police", 3),
    ],
    "MEDICAL_EXAM": [
        ("medical examination", 4), ("panel physician", 5), ("imm 1017", 5),
        ("medical report", 3), ("fit for", 2), ("chest x-ray", 2), ("medical certificate", 3),
    ],
    "MARRIAGE_CERTIFICATE": [
        ("marriage certificate", 5), ("nikah nama", 5), ("nikahnama", 5),
        ("certificate of marriage", 5), ("solemnized", 2),
    ],
    "BIRTH_CERTIFICATE": [
        ("birth certificate", 5), ("certificate of birth", 5), ("date of birth", 1),
        ("registrar of births", 4),
    ],
    "EMPLOYMENT_LETTER": [
        ("experience certificate", 5), ("to whom it may concern", 3), ("this is to certify", 2),
        ("employment certificate", 5), ("appointment letter", 4), ("offer letter", 3),
        ("service certificate", 4), ("designation", 1), ("relieving letter", 4),
    ],
    "LMIA": [
        ("labour market impact assessment", 6), ("lmia", 5), ("service canada", 3),
        ("annex a", 3), ("employment and social development canada", 4),
    ],
    "BUSINESS_PLAN": [
        ("business plan", 5), ("executive summary", 3), ("market analysis", 3),
        ("financial projections", 3), ("revenue model", 2),
    ],
    "INCORPORATION": [
        ("certificate of incorporation", 6), ("articles of association", 4),
        ("company registration", 4), ("securities and exchange commission of pakistan", 4),
        ("secp", 3), ("registrar of companies", 4),
    ],
    "TAX_RETURN": [
        ("income tax return", 5), ("tax assessment", 4), ("federal board of revenue", 4),
        ("fbr", 3), ("taxable income", 3), ("ntn", 2),
    ],
    "SPONSORSHIP_LETTER": [
        ("affidavit of support", 5), ("sponsorship", 4), ("undertaking", 2),
        ("i undertake to sponsor", 5),
    ],
    "TRAVEL_ITINERARY": [
        ("itinerary", 4), ("flight", 2), ("booking reference", 3), ("e-ticket", 4),
        ("pnr", 3), ("departure", 1), ("arrival", 1), ("airline", 2),
    ],
    "VISA": [
        ("visa", 2), ("entries", 2), ("valid until", 1), ("permit", 1), ("residence permit", 3),
    ],
    "STATEMENT_OF_PURPOSE": [
        ("statement of purpose", 6), ("study plan", 4), ("letter of motivation", 4),
    ],
}


def _classify(text: str) -> Tuple[str, float]:
    """Return (doc_type, confidence in 0..1) for a single page's text."""
    lower = (text or "").lower()
    if len(lower.strip()) < 8:
        return "OTHER", 0.0

    # Strong regex signals first — identity docs are high-value + distinctive.
    has_cnic = bool(_CNIC_RE.search(text))
    has_pk_passport = bool(_PK_PASSPORT_NO_RE.search(text))
    has_mrz = bool(_MRZ_RE.search(text))
    passport_kw = sum(1 for k in ("passport", "place of issue", "date of expiry",
                                  "given name", "surname", "country code") if k in lower)
    cnic_kw = sum(1 for k in ("national identity card", "cnic", "identity number",
                              "registrar general of pakistan") if k in lower)

    if cnic_kw >= 1 and (has_cnic or "national identity card" in lower) and not has_mrz:
        return "NATIONAL_ID", 0.95
    if passport_kw >= 1 and (has_mrz or has_pk_passport or "passport" in lower):
        # passport is the most common immigration doc — score it confidently.
        return "PASSPORT", 0.95 if (has_mrz or passport_kw >= 3) else 0.8

    # Keyword scoring across the rest of the vocab.
    best_type, best_score = "OTHER", 0
    for dtype, kws in _RULES.items():
        score = sum(w for kw, w in kws if kw in lower)
        if score > best_score:
            best_type, best_score = dtype, score

    if best_score <= 1:
        return "OTHER", 0.3 if best_score == 0 else 0.45
    # Map raw score -> confidence: 2->0.6, 4->0.75, 6+->~0.9 (capped 0.95).
    conf = min(0.95, 0.45 + 0.075 * best_score)
    return best_type, round(conf, 3)


def _page_text(page, data: bytes) -> Tuple[str, str, float]:
    """Text for one PDF page: native (free) if rich enough, else Google Vision.
    Returns (text, tier, cost_cents)."""
    s = get_settings()
    native = (page.get_text() or "").strip()
    if len(native) >= NATIVE_MIN_CHARS:
        return native, "native_pdf", 0.0
    # Scanned page -> render + Vision OCR just this page.
    try:
        png = page.get_pixmap(dpi=150).tobytes("png")
    except Exception:
        return native, "native_pdf", 0.0
    vtxt = ocr.vision_ocr_images([png])
    if vtxt:
        return vtxt, "google_vision", s.vision_cost_cents_per_page
    return native, "native_pdf", 0.0


def run_split(data: bytes, mime: str) -> Dict[str, Any]:
    """Split a (possibly multi-document) upload into categorized documents.

    Returns { documents: [{doc_type, pages, confidence, needs_review, ocrTier}],
              pageCount, truncated, costCents, engineUsed }.
    """
    is_pdf = "pdf" in (mime or "").lower() or data[:5] == b"%PDF-"
    cost = 0.0

    # ── Single image: one page, classify directly ──────────────────────────
    if not is_pdf:
        vtxt = ocr.vision_ocr_images([data]) or ""
        cost += get_settings().vision_cost_cents_per_page if vtxt else 0.0
        dtype, conf = _classify(vtxt)
        return {
            "documents": [{
                "doc_type": dtype, "pages": [0], "confidence": conf,
                "needs_review": conf < CONFIDENCE_GATE, "ocrTier": "google_vision",
            }],
            "pageCount": 1, "truncated": False,
            "costCents": round(cost, 4), "engineUsed": "google_vision",
        }

    # ── PDF: classify each page, then group consecutive same-type pages ─────
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return {"documents": [], "pageCount": 0, "truncated": False,
                "costCents": 0.0, "engineUsed": "none", "error": "Could not open PDF"}

    total_pages = doc.page_count
    scan = min(total_pages, SPLIT_MAX_PAGES)
    page_class: List[Tuple[int, str, float]] = []  # (page_idx, doc_type, conf)
    try:
        for i in range(scan):
            text, _tier, c = _page_text(doc[i], data)
            cost += c
            dtype, conf = _classify(text)
            page_class.append((i, dtype, conf))
    finally:
        doc.close()

    # Group consecutive pages of the same doc_type into one logical document.
    documents: List[Dict[str, Any]] = []
    for idx, dtype, conf in page_class:
        if documents and documents[-1]["doc_type"] == dtype and documents[-1]["pages"][-1] == idx - 1:
            documents[-1]["pages"].append(idx)
            documents[-1]["confidence"] = min(documents[-1]["confidence"], conf)
        else:
            documents.append({"doc_type": dtype, "pages": [idx], "confidence": conf})

    for d in documents:
        d["needs_review"] = d["confidence"] < CONFIDENCE_GATE
        d["ocrTier"] = "google_vision"

    return {
        "documents": documents,
        "pageCount": total_pages,
        "truncated": total_pages > scan,
        "costCents": round(cost, 4),
        "engineUsed": "google_vision",
    }
