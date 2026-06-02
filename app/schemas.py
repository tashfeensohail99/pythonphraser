"""Request/response contract shared with the NestJS backend.

Checks are plain dicts ({"code", "pass", "detail"}) rather than a model because
"pass" is a Python keyword and the shape is trivial.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ExpectedDoc(BaseModel):
    docType: Optional[str] = None
    documentKind: Literal["TEXT_DOCUMENT", "PHOTO"] = "TEXT_DOCUMENT"
    documentName: str = ""
    validityRule: Literal[
        "NONE", "MUST_NOT_EXPIRE", "MUST_BE_VALID_FOR_N_MONTHS"
    ] = "NONE"
    validityMonths: Optional[int] = None
    validityBufferDays: int = 30
    photoSpec: Optional[Dict[str, Any]] = None
    clientName: Optional[str] = None
    # Ownership reference values from the CRM client record (any may be null).
    clientDob: Optional[str] = None
    clientPassportNumber: Optional[str] = None
    clientNationalId: Optional[str] = None
    service: Optional[str] = None
    targetCountry: Optional[str] = None


class FilePayload(BaseModel):
    url: Optional[str] = None
    contentBase64: Optional[str] = None
    mimeType: str = "application/octet-stream"
    fileName: str = "document"


class ValidateRequest(BaseModel):
    caseId: str
    documentItemId: str
    versionId: str
    expected: ExpectedDoc
    file: FilePayload
    # Per-request OpenAI key from the backend's admin-managed key store
    # (single source of truth). Falls back to OPENAI_API_KEY env if absent.
    openaiApiKey: Optional[str] = None


class ValidateResponse(BaseModel):
    detectedDocType: Optional[str] = None
    confidence: float = 0.0
    extracted: Dict[str, Any] = {}
    checks: List[Dict[str, Any]] = []
    suggestedDecision: Literal["APPROVE", "REJECT", "NEEDS_REVIEW"] = "NEEDS_REVIEW"
    reasonCodes: List[str] = []
    ocrTier: str = "none"
    costCents: float = 0.0
    cacheHit: bool = False
    modelVersion: str = ""
    # P4c-2: attestation authorities whose stamp keywords were found in the OCR
    # text (e.g. ["MOFA", "HEC"]). Suggestion only — surfaced as a hint next to
    # the manual "Mark attested" control; never auto-marks attestation.
    detectedAuthorities: List[str] = []
    # P4f: dominant non-Latin script detected in OCR text
    # (e.g. "Arabic/Urdu", "Chinese/Japanese/Korean"). None = primarily Latin.
    # Suggestion only — shown as an amber "translation needed" hint on the
    # checklist. Never auto-sets translationStatus.
    detectedLanguage: Optional[str] = None
    errorMessage: Optional[str] = None


# ── Split & categorize (multi-document upload) ──────────────────────────────


class SplitRequest(BaseModel):
    file: FilePayload
    # Optional context hints (used by the backend; the parser ignores them for
    # now but they're part of the contract for program-aware classification).
    caseId: Optional[str] = None
    expectedProgram: Optional[str] = None
    expectedDocTypes: Optional[List[str]] = None


class SplitDocument(BaseModel):
    doc_type: str
    pages: List[int]
    confidence: float = 0.0
    needs_review: bool = True
    ocrTier: str = "google_vision"
    # This segment extracted into its own standalone file (base64). The backend
    # stores this as the document version. "" if extraction failed -> triage.
    fileBase64: str = ""
    mimeType: str = "application/pdf"


class SplitResponse(BaseModel):
    documents: List[SplitDocument] = []
    pageCount: int = 0
    truncated: bool = False
    costCents: float = 0.0
    engineUsed: str = "none"
    modelVersion: str = ""
    errorMessage: Optional[str] = None
