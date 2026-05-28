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
    errorMessage: Optional[str] = None
