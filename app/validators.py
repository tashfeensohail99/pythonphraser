"""Validate extracted fields against the expected requirement and decide.

build_text_checks() turns the requirement (doc-type, validity rule, client name)
plus the LLM-extracted fields into a list of pass/fail checks. decide() collapses
those into a suggestedDecision the backend can act on (REJECT on any failure,
APPROVE on all-pass + high confidence, else NEEDS_REVIEW).
"""
import re
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

from .schemas import ExpectedDoc

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y")


def _parse_date(s) -> Optional[date]:
    if not s or not isinstance(s, str):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except Exception:
            continue
    return None


def _name_match(client_name: Optional[str], doc_name: Optional[str]) -> Optional[bool]:
    if not client_name or not doc_name:
        return None
    a = {t for t in client_name.lower().replace(".", "").split() if len(t) > 1}
    b = {t for t in doc_name.lower().replace(".", "").split() if len(t) > 1}
    if not a or not b:
        return None
    overlap = len(a & b)
    return overlap >= min(2, len(a))


def _norm_id(s: Optional[str]) -> str:
    """Normalise an ID/passport number for comparison (strip spaces/punctuation)."""
    return re.sub(r"[^A-Za-z0-9]", "", s or "").upper()


def build_text_checks(
    detected: Optional[str], confidence: float, expected: ExpectedDoc, fields: dict
) -> List[dict]:
    checks: List[dict] = []

    if expected.docType:
        checks.append(
            {
                "code": "DOC_TYPE_MATCH",
                "pass": detected == expected.docType,
                "detail": f"expected {expected.docType}, detected {detected or 'unknown'}",
            }
        )

    if expected.validityRule in ("MUST_NOT_EXPIRE", "MUST_BE_VALID_FOR_N_MONTHS"):
        exp = _parse_date(fields.get("expiryDate"))
        if exp is None:
            checks.append({"code": "EXPIRY_FOUND", "pass": False, "detail": "No expiry date found"})
        else:
            today = date.today()
            buffer = timedelta(days=expected.validityBufferDays or 0)
            if expected.validityRule == "MUST_NOT_EXPIRE":
                checks.append(
                    {
                        "code": "NOT_EXPIRED",
                        "pass": exp >= today + buffer,
                        "detail": f"expires {exp.isoformat()}",
                    }
                )
            else:
                months = expected.validityMonths or 0
                need = today + timedelta(days=30 * months) + buffer
                checks.append(
                    {
                        "code": "VALID_FOR_PERIOD",
                        "pass": exp >= need,
                        "detail": f"expires {exp.isoformat()}, needs validity through ~{need.isoformat()}",
                    }
                )

    name_in_doc = fields.get("fullName") or fields.get("accountHolder")
    nm = _name_match(expected.clientName, name_in_doc)
    if nm is not None:
        checks.append(
            {
                "code": "NAME_MATCH",
                "pass": nm,
                "detail": f"client '{expected.clientName}' vs doc '{name_in_doc}'",
            }
        )

    # Ownership: date of birth (when both the client record and the document
    # carry one — typically passport / national ID / birth certificate).
    if expected.clientDob:
        doc_dob = _parse_date(fields.get("dateOfBirth"))
        exp_dob = _parse_date(expected.clientDob)
        if doc_dob and exp_dob:
            checks.append(
                {
                    "code": "DOB_MATCH",
                    "pass": doc_dob == exp_dob,
                    "detail": f"client {exp_dob.isoformat()} vs doc {doc_dob.isoformat()}",
                }
            )

    # Ownership: passport / national-ID number (strongest belongs-to-client signal).
    if expected.clientPassportNumber and fields.get("passportNumber"):
        checks.append(
            {
                "code": "PASSPORT_NO_MATCH",
                "pass": _norm_id(fields["passportNumber"]) == _norm_id(expected.clientPassportNumber),
                "detail": f"client {expected.clientPassportNumber} vs doc {fields['passportNumber']}",
            }
        )
    if expected.clientNationalId and fields.get("idNumber"):
        checks.append(
            {
                "code": "ID_NO_MATCH",
                "pass": _norm_id(fields["idNumber"]) == _norm_id(expected.clientNationalId),
                "detail": f"client {expected.clientNationalId} vs doc {fields['idNumber']}",
            }
        )

    if expected.docType == "BANK_STATEMENT" or detected == "BANK_STATEMENT":
        sd = _parse_date(fields.get("statementDate") or fields.get("documentDate"))
        if sd:
            checks.append(
                {
                    "code": "RECENT_STATEMENT",
                    "pass": sd >= date.today() - timedelta(days=95),
                    "detail": f"statement dated {sd.isoformat()} (want <= 3 months old)",
                }
            )

    return checks


def decide(checks: List[dict], confidence: float, high_conf: float) -> Tuple[str, List[str], bool]:
    """Return (suggestedDecision, reasonCodes, autoApproveEligible)."""
    failed = [c["code"] for c in checks if not c["pass"]]
    if failed:
        return ("REJECT", failed, False)
    reasons = [c["code"] for c in checks]
    if checks and confidence >= high_conf:
        return ("APPROVE", reasons, True)
    return ("NEEDS_REVIEW", reasons, False)


def build_completeness_checks(
    detected: Optional[str],
    expected: ExpectedDoc,
    completeness: Optional[dict],
    page_count: int,
) -> List[dict]:
    """Turn the LLM completeness assessment + page count into pass/fail checks.

    Completeness is doc-type aware: ID cards need front+back, passports need a
    complete (uncropped) bio page (MRZ a strong signal), bank statements need
    the full period / all pages.
    """
    checks: List[dict] = []
    if not completeness:
        return checks
    doc_type = expected.docType or detected
    note = completeness.get("note")

    if doc_type == "NATIONAL_ID":
        checks.append(
            {
                "code": "FRONT_AND_BACK",
                "pass": bool(completeness.get("hasFrontAndBack")),
                "detail": note or "Both the front and back of the ID are required",
            }
        )
    elif doc_type == "PASSPORT":
        ok = bool(completeness.get("mrzPresent")) or bool(completeness.get("appearsComplete"))
        checks.append(
            {
                "code": "PASSPORT_COMPLETE",
                "pass": ok,
                "detail": "Bio page complete (MRZ visible)"
                if ok
                else "Passport bio page may be cropped or incomplete",
            }
        )
    elif doc_type == "BANK_STATEMENT":
        checks.append(
            {
                "code": "STATEMENT_COMPLETE",
                "pass": bool(completeness.get("appearsComplete")),
                "detail": note or f"{page_count} page(s) — must cover the full required period",
            }
        )
    elif completeness.get("appearsComplete") is False:
        checks.append(
            {
                "code": "DOCUMENT_COMPLETE",
                "pass": False,
                "detail": note or "Document appears incomplete",
            }
        )
    return checks
