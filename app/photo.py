"""Photo-requirement validation (NOT extraction).

For documentKind=PHOTO the client uploads a portrait and we validate it meets
the per-category spec: sharp (not blurry), correct aspect ratio (e.g. 35x45mm),
plain background of the required colour, and exactly one face. We never pull a
face out of another document.
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

_FACE_CASCADE = None


def _decode(data: bytes):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _border_pixels(img):
    h, w = img.shape[:2]
    bt, bw = max(1, h // 10), max(1, w // 10)
    top = img[0:bt, :].reshape(-1, 3)
    bottom = img[h - bt :, :].reshape(-1, 3)
    left = img[:, 0:bw].reshape(-1, 3)
    right = img[:, w - bw :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def _count_faces(gray) -> int:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return len(faces)


def validate_photo(data: bytes, spec: Dict[str, Any]) -> Tuple[List[dict], Dict[str, Any]]:
    checks: List[dict] = []
    extracted: Dict[str, Any] = {}

    img = _decode(data)
    if img is None:
        return ([{"code": "PHOTO_UNREADABLE", "pass": False, "detail": "Could not decode image"}], extracted)

    h, w = img.shape[:2]
    extracted.update({"widthPx": int(w), "heightPx": int(h)})
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness (variance of Laplacian)
    fm = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    max_blur = float(spec.get("maxBlur", 100))
    extracted["sharpness"] = round(fm, 1)
    checks.append(
        {"code": "NOT_BLURRY", "pass": fm >= max_blur, "detail": f"sharpness={fm:.0f} (min {max_blur:.0f})"}
    )

    # Aspect ratio
    size = str(spec.get("sizeMm", "35x45")).lower()
    try:
        sw, sh = (float(x) for x in size.split("x"))
        target = sw / sh
    except Exception:
        target = 35 / 45
    ratio = w / h if h else 0
    checks.append(
        {
            "code": "ASPECT_RATIO",
            "pass": abs(ratio - target) <= 0.12,
            "detail": f"ratio={ratio:.2f} (target {target:.2f})",
        }
    )

    # Background colour + uniformity
    background = str(spec.get("background", "ANY")).upper()
    if background and background != "ANY":
        border = _border_pixels(img).astype("float32")
        mean = border.mean(axis=0)  # BGR
        uniform = float(border.std()) < 40
        b, g, r = mean
        if background == "WHITE":
            color_ok = b > 180 and g > 180 and r > 180
        elif background == "BLUE":
            color_ok = b > 120 and b > r + 25 and b > g
        else:
            color_ok = True
        checks.append(
            {
                "code": "BACKGROUND",
                "pass": bool(uniform and color_ok),
                "detail": f"bg≈rgb({r:.0f},{g:.0f},{b:.0f}) uniform={uniform}",
            }
        )

    # Exactly one face
    if spec.get("faceRequired", True):
        n = _count_faces(gray)
        extracted["faces"] = int(n)
        checks.append({"code": "SINGLE_FACE", "pass": n == 1, "detail": f"faces detected={n}"})

    return (checks, extracted)
