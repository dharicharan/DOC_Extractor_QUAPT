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

    rects = [cv2.boundingRect(c) for c in contours]
    rects = [r for r in rects if r[3] >= 0.008 * h and r[2] >= 0.012 * w]
    heights = sorted(r[3] for r in rects if r[3] < 0.06 * h)
    if heights:
        rects = _split_tall(ink, rects, heights[len(heights) // 2])

    boxes: List[Box] = []
    for x, y, bw, bh in rects:
        if bw * bh > 0.35 * w * h or bh > 0.12 * h or bw > 0.7 * w:   # photos, borders, ruled strips
            continue
        boxes.append((1000.0 * x / w, 1000.0 * y / h, 1000.0 * (x + bw) / w, 1000.0 * (y + bh) / h))
    return boxes


def _split_tall(ink, rects, median_h: int):
    """Split regions that swallowed two text lines at the blank rows between them.

    A region much taller than the typical line is cut where the ink profile (ink per pixel row)
    drops to almost nothing; each piece is then tightened to its own ink. A region with no clear
    gap is returned as it was.
    """
    import numpy as np

    out = []
    for x, y, bw, bh in rects:
        if bh <= 1.7 * median_h:
            out.append((x, y, bw, bh))
            continue
        prof = ink[y:y + bh, x:x + bw].sum(axis=1).astype(float)
        k = max(3, int(median_h // 4))
        prof = np.convolve(prof, np.ones(k) / k, mode="same")
        rows = prof > 0.12 * prof.max()
        pieces, start = [], None
        for i, on in enumerate(list(rows) + [False]):
            if on and start is None:
                start = i
            elif not on and start is not None:
                if i - start >= max(3, int(0.4 * median_h)):
                    cols = np.where(ink[y + start:y + i, x:x + bw].sum(axis=0) > 0)[0]
                    if len(cols):
                        pieces.append((x + int(cols[0]), y + start, int(cols[-1] - cols[0] + 1), i - start))
                start = None
        out.extend(pieces if len(pieces) > 1 else [(x, y, bw, bh)])
    return out


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
    if _iou(vlm_box, best) < MIN_IOU_TO_SNAP or not _plausible(vlm_box, best):
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
    grown = (x1, y1, x2, y2)
    return grown if _plausible(vlm_box, grown) else best


def _plausible(vlm_box: Box, snapped: Box) -> bool:
    """A snap may tighten or shift a box, not blow it up: reject much wider or taller results."""
    vw, vh = vlm_box[2] - vlm_box[0], vlm_box[3] - vlm_box[1]
    sw, sh = snapped[2] - snapped[0], snapped[3] - snapped[1]
    return sw <= 1.6 * vw + 5.0 and sh <= 2.0 * vh + 5.0


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
    items_out: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Snap boxes onto detected text and verify values against a crop reading. Mutates ``raw``.

    ``read_crop(pil_crop) -> text`` is the model call; without it only the boxes are snapped.
    Returns counts for the run record: ``snapped``, ``verified``, ``agreed``, ``disagreed``.
    """
    stats: Dict[str, Any] = {"snapped": 0, "verified": 0, "agreed": 0, "disagreed": 0, "detected": 0, "errors": 0}
    claimed: set = set()
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
        if new_box != box and tuple(round(v, 1) for v in new_box) in claimed:
            new_box = box          # another field already took this line: keep this field's own box
        if new_box != box:
            claimed.add(tuple(round(v, 1) for v in new_box))
            leaf["bbox"] = [round(v, 1) for v in new_box]
            stats["snapped"] += 1
        if read_crop is None or stats["verified"] >= budget or isinstance(value, (dict, list)):
            continue
        try:
            crop = crop_region(image, new_box)
            reading = read_crop(crop)
        except Exception as exc:  # noqa: BLE001
            logger.exception("crop-verify: crop read failed for %s", path)
            stats["errors"] += 1
            stats.setdefault("first_error", f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        stats["verified"] += 1
        agree = texts_agree(value, reading)
        if items_out is not None:
            items_out.append({"field": path, "value": str(value), "crop": crop, "crop_text": reading, "agree": agree})
        if agree:
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


# --------------------------------------------------------------------------- demo display

_RUN_ITEMS: List[Dict[str, Any]] = []
_RUN_LOCK = __import__("threading").Lock()


_RUN_PAGES = [0]


def begin_run() -> None:
    """Forget the crops of the previous document (the demo handles one document at a time)."""
    with _RUN_LOCK:
        _RUN_ITEMS.clear()
        _RUN_PAGES[0] = 0


def record(items: Sequence[Dict[str, Any]]) -> None:
    """Keep one page's crops for display; pages are numbered in the order they are read."""
    with _RUN_LOCK:
        _RUN_PAGES[0] += 1
        _RUN_ITEMS.extend({**it, "page": _RUN_PAGES[0]} for it in items)


def collect() -> List[Dict[str, Any]]:
    with _RUN_LOCK:
        return list(_RUN_ITEMS)


def _cell(text: Any) -> str:
    import html

    return html.escape(str(text)).replace("|", "&#124;").replace("\n", " ")


def _data_uri(img, max_w: int = 300, max_h: int = 64) -> str:
    import base64
    import io

    thumb = img.copy()
    thumb.thumbnail((max_w, max_h))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_markdown(items: Sequence[Dict[str, Any]]) -> str:
    """One row per re-read field: the cropped image, the page reading and the crop's OCR text."""
    if not items:
        return ""
    agreed = sum(1 for it in items if it["agree"])
    rows = [
        f"\n\n### Crop check ({agreed} of {len(items)} fields match their crop reading)\n",
        "| Page | Field | Crop | Page reading | Crop OCR text | Match |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for it in items:
        rows.append(
            f"| {it.get('page', 1)} | {_cell(it['field'])} | <img src=\"{_data_uri(it['crop'])}\" height=\"48\"> "
            f"| {_cell(it['value'])} | {_cell(it['crop_text'])} | {'✅' if it['agree'] else '❌ review'} |"
        )
    return "\n".join(rows)
