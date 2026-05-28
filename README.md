# Tashfeen Immigration Document Parser

A small FastAPI service that the Tashfeen CRM backend calls to OCR + validate a
client-uploaded immigration document against the requirement it was uploaded
for. It returns a structured assessment (detected type, extracted fields,
pass/fail checks, and a suggested decision) that the backend stores as a
`DocumentAiAssessment` and either surfaces to an associate (shadow) or acts on
(auto-approve when confidence is high and every check passes).

## Design goals
- **Cost first.** A tiered OCR ladder spends the least money that works:
  1. native PDF text (free)
  2. Google Vision OCR (~0.15¢/page, capped at `MAX_VISION_PAGES`)
  3. gpt-4o-mini vision (last resort only)
  A SHA-256 content cache means a re-uploaded identical file costs nothing.
- **No AWS.** Google Vision only.
- **No image extraction.** For photo requirements the client uploads a portrait
  and we *validate* it (background colour, 35×45 size, single face, sharpness).
  We never pull a face out of another document.
- **Fails open, never crashes.** Missing OpenAI/Vision credentials degrade the
  pipeline and return `NEEDS_REVIEW` rather than erroring.

## Endpoints
- `GET /health` — liveness + which integrations are configured.
- `POST /validate-document` — HMAC-protected. See contract below.

### Auth
The backend signs the raw request body: `X-Signature: hex(HMAC_SHA256(PYTHON_HMAC_SECRET, body))`.
The service fails closed if `PYTHON_HMAC_SECRET` is unset.

### Request
```json
{
  "caseId": "…",
  "documentItemId": "…",
  "versionId": "…",
  "expected": {
    "docType": "PASSPORT",
    "documentKind": "TEXT_DOCUMENT",
    "documentName": "Passport",
    "validityRule": "MUST_BE_VALID_FOR_N_MONTHS",
    "validityMonths": 6,
    "validityBufferDays": 30,
    "photoSpec": null,
    "clientName": "Ali Haider",
    "service": "STUDY_VISA",
    "targetCountry": "CA"
  },
  "file": { "url": "https://…signed…", "mimeType": "application/pdf", "fileName": "passport.pdf" }
}
```
Provide either `file.url` (a signed URL we fetch) or `file.contentBase64`.

For a photo requirement set `documentKind: "PHOTO"` and pass `photoSpec`, e.g.
`{ "background": "WHITE", "sizeMm": "35x45", "faceRequired": true, "maxBlur": 100 }`.

### Response
```json
{
  "detectedDocType": "PASSPORT",
  "confidence": 0.96,
  "extracted": { "fullName": "ALI HAIDER", "expiryDate": "2029-03-01" },
  "checks": [
    { "code": "DOC_TYPE_MATCH", "pass": true, "detail": "expected PASSPORT, detected PASSPORT" },
    { "code": "VALID_FOR_PERIOD", "pass": true, "detail": "expires 2029-03-01, …" },
    { "code": "NAME_MATCH", "pass": true, "detail": "client 'Ali Haider' vs doc 'ALI HAIDER'" }
  ],
  "suggestedDecision": "APPROVE",
  "reasonCodes": ["DOC_TYPE_MATCH", "VALID_FOR_PERIOD", "NAME_MATCH"],
  "ocrTier": "native_pdf",
  "costCents": 0.0,
  "cacheHit": false,
  "modelVersion": "imm-parser-1.0.0",
  "errorMessage": null
}
```

## Configuration
See `.env.example`. Only `PYTHON_HMAC_SECRET` is effectively required.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in secrets
uvicorn app.main:app --reload
```

## Deploy (Railway)
Dockerfile-based (`railway.json` points at it; healthcheck `/health`). Set the
env vars from `.env.example` in the Railway service. opencv's system libs
(`libgl1`, `libglib2.0-0`) are installed by the Dockerfile.
