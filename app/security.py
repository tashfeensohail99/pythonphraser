"""HMAC-SHA256 request authentication.

The backend computes hex(HMAC_SHA256(secret, raw_request_body)) and sends it in
the X-Signature header. We recompute over the exact bytes and constant-time
compare. Starlette caches request.body(), so reading it here doesn't stop the
route from parsing the same body into the Pydantic model.
"""
import hashlib
import hmac

from fastapi import Header, HTTPException, Request, status

from .config import get_settings


async def verify_hmac(request: Request, x_signature: str = Header(default="")) -> None:
    secret = get_settings().python_hmac_secret
    if not secret:
        # Fail closed — never run unauthenticated.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "PYTHON_HMAC_SECRET not configured",
        )
    body = await request.body()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not x_signature or not hmac.compare_digest(expected, x_signature):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing signature")
