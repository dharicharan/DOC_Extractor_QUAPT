"""Crop-and-verify pass for the vision route (prototype, off unless DOC_CROP_VERIFY=1).

The VLM's boxes are estimates: on a handwritten form they drift by a row and are spaced on an even
grid. This module improves them with plain OpenCV, then double-checks each value:

1. ``detect_text_boxes``  finds real text regions (form lines removed, words joined into lines).
2. ``snap_bbox``          moves each VLM box onto the detected region it overlaps most.
3. ``refine_raw``         crops that region, has the model read the crop on its own, and compares the
                          crop reading with the full-page value. Agreement keeps the confidence; a
                          disagreement halves it and marks the field uncertain, so it goes to review.

Boxes are ``[x1, y1, x2, y2]`` on the 0-1000 grid used everywhere else. Every step fails safe: on any
error the raw model output is returned untouched.
"""
from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

Box = Tuple[float, float, float, float]

AGREE_RATIO = 0.85          # similarity at or above this counts as "same text"
DISAGREE_FACTOR = 0.5       # confidence multiplier when the crop reading disagrees
MIN_IOU_TO_SNAP = 0.10      # below this the VLM box is kept as it was
_MAX_DETECT_PIXELS = 2_000_000


def enabled() -> bool:
    return os.environ.get("DOC_CROP_VERIFY", "").strip().lower() in {"1", "true", "yes", "on"}


def max_crops() -> int:
    try:
        return max(0, int(os.environ.get("DOC_CROP_VERIFY_MAX", "40")))
    except ValueError:
        return 40


# --------------------------------------------------------------------------- detection

def detect_text_boxes(image) -> List[Box]:
    """Text-line regions of a PIL image as 0-1000 boxes. Form rules, borders and big graphics are dropped."""
    import cv2
    import numpy as np

    rgb = np.asarray(image.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    if w0 * h0 > _MAX_DETECT_PIXELS:
        s = (_MAX_DETECT_PIXELS / float(w0 * h0)) ** 0.5
        rgb = cv2.resize(rgb, (max(1, int(w0 * s)), max(1, int(h0 * s))), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    block = max(15, (min(h, w) // 40) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15)

    # Remove long horizontal / vertical rules (table lines, underlines, borders).
    hlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 12), 1)))
    vlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 12))))
    ink = cv2.subtract(binary, cv2.dilate(cv2.bitwise_or(hlines, vlines), np.ones((3, 3), np.uint8)))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop speckle

    # Join letters into words and words into one line (wide, flat kernel).
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 70), max(3, h // 450))))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Box] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < 0.008 * h or bw < 0.012 * w:
            continue
        if bw * bh > 0.35 * w * h or bh > 0.12 * h:   # photos, page borders, merged blocks
            continue
        boxes.append((1000.0 * x / w, 1000.0 * y / h, 1000.0 * (x + bw) / w, 1000.0 * (y + bh) / h))
    return boxes


# --------------------------------------------------------------------------- snapping

def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a: Box, b: Box) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _same_line(a: Box, b: Box) -> bool:
    ov = min(a[3], b[3]) - max(a[1], b[1])
    return ov >= 0.6 * min(a[3] - a[1], b[3] - b[1])


def snap_bbox(vlm_box: Box, detected: Sequence[Box]) -> Box:
    """Move a VLM box onto the detected text line it overlaps most; keep it if nothing matches."""
    if not detected:
        return vlm_box
    best = max(detected, key=lambda d: _iou(vlm_box, d))
    if _iou(vlm_box, best) < MIN_IOU_TO_SNAP:
        return vlm_box
    # Grow along the same line to cover neighbouring words that sit inside the VLM box's width.
    x1, y1, x2, y2 = best
    reach = (vlm_box[2] - vlm_box[0]) * 0.15
    lo, hi = vlm_box[0] - reach, vlm_box[2] + reach
    height = best[3] - best[1]
    grew = True
    while grew:
        grew = False
        for d in detected:
            if d == best or not _same_line((x1, y1, x2, y2), d):
                continue
            gap = max(d[0] - x2, x1 - d[2])
            if gap <= 1.2 * height and d[2] >= lo and d[0] <= hi and not (x1 <= d[0] and d[2] <= x2):
                nx1, nx2 = min(x1, d[0]), max(x2, d[2])
                if (nx1, nx2) != (x1, x2):
                    x1, x2, grew = nx1, nx2, True
    return (x1, y1, x2, y2)


# --------------------------------------------------------------------------- verification

def _norm_text(s: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(s).lower())


def texts_agree(a: Any, b: Any) -> bool:
    """True when two readings are the same text, ignoring case, spacing and punctuation."""
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return na == nb or min(len(na), len(nb)) / max(len(na), len(nb)) >= AGREE_RATIO
    return SequenceMatcher(None, na, nb).ratio() >= AGREE_RATIO


def crop_region(image, box: Box, pad: float = 0.25, min_side: int = 64):
    """Crop a 0-1000 box (with padding) from a PIL image; tiny crops are scaled up for the model."""
    from PIL import Image

    w, h = image.size
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * 0.04, (y2 - y1) * pad
    left, top = max(0, int((x1 - px) * w / 1000.0)), max(0, int((y1 - py) * h / 1000.0))
    right, bottom = min(w, int((x2 + px) * w / 1000.0) + 1), min(h, int((y2 + py) * h / 1000.0) + 1)
    crop = image.crop((left, top, max(right, left + 1), max(bottom, top + 1))).convert("RGB")
    cw, ch = crop.size
    scale = max(1.0, min_side / float(min(cw, ch)), 3.0 if ch < 40 else 1.0)
    if scale > 1.0:
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    return crop


def _leaves(raw: Any, prefix: str = "") -> List[Tuple[str, Dict[str, Any]]]:
    """Leaf dicts (those holding a ``value`` key) inside the raw model JSON, mutable in place."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        if "value" in raw:
            out.append((prefix, raw))
        else:
            for k, v in raw.items():
                out.extend(_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            out.extend(_leaves(v, f"{prefix}.{i}" if prefix else str(i)))
    return out


def _as_box(b: Any) -> Optional[Box]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in b)
    except (TypeError, ValueError):
        return None
    x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1 or max(x2, y2) > 1000.0 or min(x1, y1) < 0.0:
        return None
    return x1, y1, x2, y2


def refine_raw(
    raw: Dict[str, Any],
    image,
    read_crop: Optional[Callable[[Any], str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Snap boxes onto detected text and verify values against a crop reading. Mutates ``raw``.

    ``read_crop(pil_crop) -> text`` is the model call; without it only the boxes are snapped.
    Returns counts for the run record: ``snapped``, ``verified``, ``agreed``, ``disagreed``.
    """
    stats = {"snapped": 0, "verified": 0, "agreed": 0, "disagreed": 0, "detected": 0}
    try:
        detected = detect_text_boxes(image)
    except Exception:  # noqa: BLE001 - never break extraction because of the optional pass
        logger.exception("crop-verify: text detection failed; boxes left as the model gave them")
        return stats
    stats["detected"] = len(detected)
    budget = max_crops() if limit is None else limit

    for path, leaf in _leaves(raw):
        value = leaf.get("value")
        box = _as_box(leaf.get("bbox"))
        if value is None or box is None:
            continue
        new_box = snap_bbox(box, detected)
        if new_box != box:
            leaf["bbox"] = [round(v, 1) for v in new_box]
            stats["snapped"] += 1
        if read_crop is None or stats["verified"] >= budget or isinstance(value, (dict, list)):
            continue
        try:
            reading = read_crop(crop_region(image, new_box))
        except Exception:  # noqa: BLE001
            logger.exception("crop-verify: crop read failed for %s", path)
            continue
        stats["verified"] += 1
        if texts_agree(value, reading):
            stats["agreed"] += 1
            leaf["crop_verified"] = True
        else:
            stats["disagreed"] += 1
            leaf["crop_verified"] = False
            leaf["crop_text"] = reading
            leaf["uncertain"] = True
            conf = leaf.get("confidence")
            if isinstance(conf, (int, float)) and not isinstance(conf, bool):
                leaf["confidence"] = round(float(conf) * DISAGREE_FACTOR, 4)
    return stats
