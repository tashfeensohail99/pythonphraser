"""Validate extracted fields against the expected requirement and decide.

build_text_checks() turns the requirement (doc-type, validity rule, client name)
plus the LLM-extracted fields into a list of pass/fail checks. decide() collapses
those into a suggestedDecision the backend can act on (REJECT on any failure,
APPROVE on all-pass + high confidence, else NEEDS_REVIEW).
"""
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
