"""FastAPI entrypoint — the /validate-document orchestrator.

Flow:
  load file (url or base64)
    -> SHA-256 cache hit? return it
    -> PHOTO requirement?  photo.validate_photo (no OCR/LLM)
    -> TEXT document:       OCR ladder -> LLM classify/extract -> validate -> decide
  cache + return.
"""
import base64

import httpx
from fastapi import Depends, FastAPI, HTTPException

from . import llm, ocr, photo, split, validators
from .cache import content_hash, get_cache
from .config import get_settings
from .schemas import (
    SplitRequest,
    SplitResponse,
    ValidateRequest,
    ValidateResponse,
)
from .security import verify_hmac

MODEL_VERSION = "imm-parser-1.3.0"

app = FastAPI(title="Tashfeen Immigration Document Parser", version=MODEL_VERSION)


@app.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "model": MODEL_VERSION,
        "openai": bool(s.openai_api_key),
        "vision": s.vision_configured(),
        "redis": bool(s.redis_url),
    }


@app.get("/health/vision")
def health_vision():
    """Active end-to-end Google Vision check.

    /health only reports whether the credential VAR is set; the OCR path then
    swallows any Vision error (returns None), so a bad key / disabled API / no
    billing looks identical to "no text". This endpoint runs a REAL
    document_text_detection on a generated text image and surfaces the true
    verdict + the exact error, so "is OCR actually working?" has a real answer.
    """
    s = get_settings()
    if not s.vision_configured():
        return {
            "configured": False, "clientInit": False, "ok": False,
            "error": "GOOGLE_VISION_CREDENTIALS_JSON / GOOGLE_APPLICATION_CREDENTIALS not set",
        }
    client = ocr._vision_client()
    if client is None:
        return {
            "configured": True, "clientInit": False, "ok": False,
            "error": "client init failed — credentials JSON invalid or unparseable",
        }
    try:
        import io

        from google.cloud import vision
        from PIL import Image, ImageDraw

        # Generate a high-contrast text image (upscaled so the built-in bitmap
        # font is large enough for Vision to read reliably).
        base = Image.new("RGB", (260, 50), "white")
        ImageDraw.Draw(base).text((8, 16), "TASHFEEN VISION OK 2468", fill="black")
        img = base.resize((1040, 200), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        resp = client.document_text_detection(image=vision.Image(content=buf.getvalue()))
        if resp.error.message:
            return {"configured": True, "clientInit": True, "ok": False, "error": resp.error.message}
        text = (resp.full_text_annotation.text or "").strip()
        return {"configured": True, "clientInit": True, "ok": True, "sampleText": text[:120]}
    except Exception as e:  # PermissionDenied / API-not-enabled / billing / etc.
        return {"configured": True, "clientInit": True, "ok": False, "error": f"{type(e).__name__}: {e}"}


async def _load_file(f) -> bytes:
    s = get_settings()
    if f.contentBase64:
        try:
            return base64.b64decode(f.contentBase64)
        except Exception:
            raise HTTPException(400, "Invalid base64 content")
    if f.url:
        try:
            async with httpx.AsyncClient(
                timeout=s.http_timeout_seconds, follow_redirects=True
            ) as client:
                r = await client.get(f.url)
                r.raise_for_status()
                return r.content
        except HTTPException:
            raise
        except Exception as e:  # network / 4xx / 5xx
            raise HTTPException(502, f"Could not fetch file.url: {e}")
    raise HTTPException(400, "file.url or file.contentBase64 is required")


@app.post(
    "/validate-document",
    response_model=ValidateResponse,
    dependencies=[Depends(verify_hmac)],
)
async def validate_document(req: ValidateRequest):
    s = get_settings()
    cache = get_cache()

    data = await _load_file(req.file)
    if len(data) > s.max_file_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_file_mb} MB limit")

    key = content_hash(
        data, req.expected.docType or "", req.expected.documentKind, MODEL_VERSION
    )
    cached = cache.get(key)
    if cached:
        cached["cacheHit"] = True
        return cached

    # ---- PHOTO requirement: validate the portrait; no OCR / LLM ----
    if req.expected.documentKind == "PHOTO":
        checks, extracted = photo.validate_photo(data, req.expected.photoSpec or {})
        suggested, reasons, _auto = validators.decide(checks, 1.0, s.high_confidence)
        resp = ValidateResponse(
            detectedDocType="PHOTOGRAPH",
            confidence=1.0,
            extracted=extracted,
            checks=checks,
            suggestedDecision=suggested,
            reasonCodes=reasons,
            ocrTier="photo_only",
            costCents=0.0,
            modelVersion=MODEL_VERSION,
        ).model_dump()
        cache.set(key, resp)
        return resp

    # ---- TEXT document: OCR ladder -> LLM classify/extract -> validate ----
    cost = 0.0
    error = None

    result = ocr.extract_text(data, req.file.mimeType)
    cost += result.cost
    text, tier = result.text, result.tier

    if tier == "needs_vision_fallback":
        is_pdf = "pdf" in (req.file.mimeType or "").lower() or data[:5] == b"%PDF-"
        images = ocr.render_pdf_pages_png(data, s.max_vision_pages) if is_pdf else [data]
        text, vision_cost = llm.vision_ocr_llm_images(images, req.openaiApiKey)
        cost += vision_cost
        tier = "gpt4o_mini_vision"

    detected = None
    confidence = 0.0
    fields: dict = {}
    completeness: dict | None = None

    if text:
        parsed, llm_cost = llm.classify_and_extract(text, req.expected, req.openaiApiKey)
        cost += llm_cost
        if parsed:
            detected = parsed.get("detectedDocType")
            try:
                confidence = float(parsed.get("confidence") or 0.0)
            except Exception:
                confidence = 0.0
            fields = {k: v for k, v in (parsed.get("fields") or {}).items() if v}
            completeness = parsed.get("completeness")
        else:
            error = "OpenAI not configured or returned no data — manual review needed"
    else:
        error = "No text could be extracted from the document"

    pages = ocr.page_count(data, req.file.mimeType)
    if pages:
        fields = {**fields, "pageCount": pages}
    checks = validators.build_text_checks(detected, confidence, req.expected, fields)
    checks += validators.build_completeness_checks(detected, req.expected, completeness, pages)
    if error and not checks:
        suggested, reasons = "NEEDS_REVIEW", []
    else:
        suggested, reasons, _auto = validators.decide(checks, confidence, s.high_confidence)

    resp = ValidateResponse(
        detectedDocType=detected,
        confidence=round(confidence, 3),
        extracted=fields,
        checks=checks,
        suggestedDecision=suggested,
        reasonCodes=reasons,
        ocrTier=tier,
        costCents=round(cost, 4),
        modelVersion=MODEL_VERSION,
        # P4c-2: attestation-stamp hint from the OCR text (suggestion only).
        detectedAuthorities=validators.detect_authorities(text),
        errorMessage=error,
    ).model_dump()
    cache.set(key, resp)
    return resp


@app.post(
    "/split-and-categorize",
    response_model=SplitResponse,
    dependencies=[Depends(verify_hmac)],
)
async def split_and_categorize(req: SplitRequest):
    """Split a (possibly multi-document) upload into categorized documents.

    OCR-first (native PDF text / Google Vision) + deterministic classification
    over the canonical DOC_TYPES vocab. High-confidence segments are trusted;
    low-confidence ones come back needs_review=True for the associate's
    Split Reviewer. No OpenAI vision in this path.
    """
    s = get_settings()
    data = await _load_file(req.file)
    if len(data) > s.max_file_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_file_mb} MB limit")

    result = split.run_split(data, req.file.mimeType)
    return SplitResponse(
        documents=result.get("documents", []),
        pageCount=result.get("pageCount", 0),
        truncated=result.get("truncated", False),
        costCents=result.get("costCents", 0.0),
        engineUsed=result.get("engineUsed", "none"),
        modelVersion=MODEL_VERSION,
        errorMessage=result.get("error"),
    )
