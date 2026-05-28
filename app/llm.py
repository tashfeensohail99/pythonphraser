"""OpenAI calls — classification/extraction (Structured Outputs) and a
last-resort vision OCR. gpt-4o-mini throughout; temperature 0 for determinism.

Returns ({}, 0.0) when no API key is configured so the caller degrades to
NEEDS_REVIEW rather than crashing.
"""
import base64
import json
from typing import List, Tuple

from .config import get_settings
from .schemas import ExpectedDoc
from .vocab import DOC_TYPES, FIELD_KEYS


def _client():
    s = get_settings()
    if not s.openai_api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=s.openai_api_key)
    except Exception:
        return None


def _cost(resp, _model: str) -> float:
    """Estimate cents for gpt-4o-mini ($0.15/1M in, $0.60/1M out)."""
    try:
        u = resp.usage
        return (u.prompt_tokens or 0) * 0.000015 + (u.completion_tokens or 0) * 0.00006
    except Exception:
        return 0.0


def _strict_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["detectedDocType", "confidence", "fields", "completeness"],
        "properties": {
            "detectedDocType": {"type": "string", "enum": list(DOC_TYPES.keys())},
            "confidence": {"type": "number"},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "required": FIELD_KEYS,
                "properties": {k: {"type": ["string", "null"]} for k in FIELD_KEYS},
            },
            "completeness": {
                "type": "object",
                "additionalProperties": False,
                "required": ["appearsComplete", "hasFrontAndBack", "mrzPresent", "note"],
                "properties": {
                    "appearsComplete": {"type": "boolean"},
                    "hasFrontAndBack": {"type": "boolean"},
                    "mrzPresent": {"type": "boolean"},
                    "note": {"type": ["string", "null"]},
                },
            },
        },
    }


def classify_and_extract(text: str, expected: ExpectedDoc) -> Tuple[dict, float]:
    client = _client()
    if client is None:
        return ({}, 0.0)
    s = get_settings()
    system = (
        "You are an immigration document analyst. Classify the document into one "
        "of the allowed types and extract the requested fields. Use ISO-8601 "
        "(YYYY-MM-DD) for every date. Return null for any field you cannot find. "
        "Be conservative: only give high confidence when the document is "
        "unambiguous.\n"
        "Always extract fullName, dateOfBirth, passportNumber and idNumber when "
        "shown — they confirm the document belongs to the right person.\n"
        "Also assess completeness:\n"
        "- appearsComplete: is the document whole and untruncated? A bank "
        "statement must show ALL pages and the full period, not just page 1.\n"
        "- hasFrontAndBack: for national ID / CNIC cards, are BOTH the front and "
        "back present?\n"
        "- mrzPresent: for passports, is the 2-line machine-readable zone at the "
        "bottom of the bio page visible (a strong sign the page is complete and "
        "not cropped)?\n"
        "Set booleans to false when not applicable or not determinable."
    )
    hint = ""
    if expected.docType and expected.docType in DOC_TYPES:
        spec = DOC_TYPES[expected.docType]
        hint = (
            f"This document is expected to be {expected.docType} ({spec['desc']}). "
            f"Fields of interest: {', '.join(spec['fields']) or 'n/a'}.\n\n"
        )
    user = f"{hint}DOCUMENT TEXT:\n{text[:12000]}"
    try:
        resp = client.chat.completions.create(
            model=s.openai_model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "doc_assessment", "strict": True, "schema": _strict_schema()},
            },
        )
        data = json.loads(resp.choices[0].message.content)
        return (data, _cost(resp, s.openai_model))
    except Exception:
        return ({}, 0.0)


def vision_ocr_llm_images(images: List[bytes]) -> Tuple[str, float]:
    """Last-resort OCR: ask gpt-4o-mini vision to transcribe page images."""
    client = _client()
    if client is None or not images:
        return ("", 0.0)
    s = get_settings()
    content: List[dict] = [
        {"type": "text", "text": "Transcribe ALL text visible in these document image(s) verbatim."}
    ]
    for img in images[: s.max_vision_pages]:
        b64 = base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    try:
        resp = client.chat.completions.create(
            model=s.openai_vision_model,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        return (resp.choices[0].message.content or "", _cost(resp, s.openai_vision_model))
    except Exception:
        return ("", 0.0)
