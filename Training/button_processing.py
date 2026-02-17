"""
button_processing — General-purpose UI button detector.

Combines YOLOv8n with CV-based heuristics to locate clickable UI elements
(buttons, tabs, toolbar items, etc.) on **any** screenshot.

Public API
----------
    detect_buttons(img)          → list[dict]   (JSON-serialisable)
    detect_buttons_to_json(img)  → str          (pretty-printed JSON string)
    annotate_buttons(img, buttons) → np.ndarray  (image with boxes drawn)

CLI
---
    python button_processing.py <image> [--out-image FILE] [--out-json FILE]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ═══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: List[Box], scores: List[float],
         thresh: float = 0.35) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    suppressed: set[int] = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j in suppressed or j == i:
                continue
            if _iou(boxes[i], boxes[j]) > thresh:
                suppressed.add(j)
    return keep


def _region_contrast(img: np.ndarray,
                     x1: int, y1: int, x2: int, y2: int,
                     margin: int = 15) -> float:
    """Colour difference between the ROI and its surrounding ring (0→1)."""
    h, w = img.shape[:2]
    roi = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    if roi.size == 0:
        return 0.0
    ox1, oy1 = max(x1 - margin, 0), max(y1 - margin, 0)
    ox2, oy2 = min(x2 + margin, w), min(y2 + margin, h)
    outer = img[oy1:oy2, ox1:ox2]
    if outer.size == 0:
        return 0.0
    diff = np.linalg.norm(
        roi.astype(float).mean(axis=(0, 1))
        - outer.astype(float).mean(axis=(0, 1))
    )
    return min(diff / 80.0, 1.0)


def _is_code_or_prose(text: str) -> bool:
    t = text.strip()
    if "  " in t:
        return True
    for tok in ("(", ")", "=", "{", "}", "//", "/*", "*/",
                "import ", "def ", "class ", "return ", "print(",
                "for ", "while ", "#include", "var ", "let ", "const ",
                "->", "=>", "::", "&&", "||", "<<", ">>"):
        if tok in t:
            return True
    if len(t) > 40 and t.count(" ") >= 5:
        return True
    if "/" in t and len(t) > 20:
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — YOLOv8n
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_yolo(img_path: str | None,
                 img: np.ndarray | None = None,
                 ) -> Tuple[List[Box], List[float], List[str]]:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    source = img_path if img_path else img
    results = model(source, conf=0.30, verbose=False)
    boxes, scores, labels = [], [], []
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            scores.append(float(b.conf[0]))
            labels.append(model.names[int(b.cls[0])])
    return boxes, scores, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — OCR-guided
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ocr(img: np.ndarray) -> dict:
    """Run OCR, using the project engine if available, else plain RapidOCR."""
    try:
        parent = str(Path(__file__).resolve().parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from image_processing import run_ocr
        return run_ocr(img)
    except Exception:
        pass
    from rapidocr import RapidOCR
    engine = RapidOCR()
    result = engine(img)
    if not result.txts:
        return {"texts": [], "boxes": [], "scores": []}
    return {
        "texts": list(result.txts),
        "boxes": [b.tolist() for b in result.boxes],
        "scores": list(result.scores),
    }


def _detect_ocr_buttons(img: np.ndarray,
                         ) -> Tuple[List[Box], List[float], List[str]]:
    data = _run_ocr(img)
    img_h, img_w = img.shape[:2]
    PAD_X, PAD_Y = 6, 4
    boxes, scores, labels = [], [], []

    for text, poly, _ in zip(data["texts"], data["boxes"], data["scores"]):
        t = text.strip()
        pts = np.array(poly)
        x1, y1 = pts.min(axis=0).astype(int)
        x2, y2 = pts.max(axis=0).astype(int)
        bw, bh = x2 - x1, y2 - y1

        if len(t) <= 1 or len(t) > 35:
            continue
        if bw > img_w * 0.35 or bh > 50:
            continue
        if t.replace(".", "").replace(",", "").isdigit() and bw < 50:
            continue
        if _is_code_or_prose(t):
            continue

        contrast = _region_contrast(img, x1, y1, x2, y2)
        if contrast < 0.12:
            continue

        conf = 0.35
        if len(t) <= 15:
            conf += 0.08
        if len(t) <= 8:
            conf += 0.05
        conf += 0.30 * contrast
        if t == t.upper() and len(t) >= 3 and t.isalpha():
            conf += 0.12
        if bh > 0 and bw / bh >= 1.5:
            conf += 0.05
        if conf < 0.60:
            continue

        boxes.append((max(x1 - PAD_X, 0), max(y1 - PAD_Y, 0),
                      min(x2 + PAD_X, img_w), min(y2 + PAD_Y, img_h)))
        scores.append(min(conf, 0.95))
        labels.append(t)

    return boxes, scores, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 3 — Accent-colour blobs
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_accent(img: np.ndarray,
                   ) -> Tuple[List[Box], List[float], List[str]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_h, img_w = img.shape[:2]
    mask = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 45)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
                            iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores, labels = [], [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if w < 25 or h < 12 or area < 400:
            continue
        if w > img_w * 0.35 or h > 70:
            continue
        if h > 0 and w / h < 0.8:
            continue
        fill = mask[y:y + h, x:x + w].sum() / 255 / area if area else 0
        if fill < 0.35:
            continue
        boxes.append((x, y, x + w, y + h))
        scores.append(min(0.55 + 0.30 * fill, 0.90))
        labels.append(None)

    return boxes, scores, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Strategy 4 — Rectangular contours
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_rects(img: np.ndarray,
                  ) -> Tuple[List[Box], List[float], List[str]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = img.shape[:2]
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 40, 130)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes, scores, labels = [], [], []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 80:
            continue
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if not (4 <= len(approx) <= 6):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 16 or h > 55 or w > img_w * 0.30:
            continue
        if h > 0 and w / h < 1.3:
            continue
        cnt_area = cv2.contourArea(cnt)
        rect_area = w * h
        if rect_area == 0:
            continue
        rr = cnt_area / rect_area
        if rr < 0.70:
            continue
        contrast = _region_contrast(img, x, y, x + w, y + h)
        if contrast < 0.06:
            continue
        boxes.append((x, y, x + w, y + h))
        scores.append(min(0.40 + 0.25 * rr + 0.25 * contrast, 0.85))
        labels.append(None)

    return boxes, scores, labels


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_buttons(img: np.ndarray,
                   img_path: str | None = None,
                   ) -> list[dict]:
    """Detect UI buttons on a screenshot.

    Args:
        img:      BGR numpy array (OpenCV format).
        img_path: Optional path to the image file on disk.  If provided the
                  YOLO strategy can read the file directly (slightly faster).

    Returns:
        A list of dicts, sorted by confidence (highest first)::

            [
                {
                    "id":         0,
                    "bbox":       {"x1": 10, "y1": 20, "x2": 120, "y2": 48},
                    "center":     {"x": 65, "y": 34},
                    "size":       {"width": 110, "height": 28},
                    "confidence": 0.87,
                    "source":     "ocr",
                    "label":      "Submit"    # may be null
                },
                ...
            ]
    """
    if img is None or img.size == 0:
        return []

    all_boxes:  list[Box]        = []
    all_scores: list[float]      = []
    all_labels: list[str | None] = []
    all_srcs:   list[str]        = []

    strategies: list[tuple[str, ...]] = [
        ("yolo",),
        ("ocr",),
        ("accent",),
        ("contour",),
    ]
    _fns = {
        "yolo":    lambda: _detect_yolo(img_path, img),
        "ocr":     lambda: _detect_ocr_buttons(img),
        "accent":  lambda: _detect_accent(img),
        "contour": lambda: _detect_rects(img),
    }

    for (name,) in strategies:
        try:
            b, s, l = _fns[name]()
        except Exception as exc:
            logger.warning("Strategy %s failed: %s", name, exc)
            continue
        all_boxes.extend(b)
        all_scores.extend(s)
        all_labels.extend(l)
        all_srcs.extend([name] * len(b))

    if not all_boxes:
        return []

    # ── NMS ──────────────────────────────────────────────────────────────
    keep = _nms(all_boxes, all_scores, thresh=0.30)
    buttons: list[dict] = []
    for idx, k in enumerate(keep):
        x1, y1, x2, y2 = all_boxes[k]
        w = x2 - x1
        h = y2 - y1
        buttons.append({
            "id":         idx,
            "bbox":       {"x1": int(x1), "y1": int(y1),
                           "x2": int(x2), "y2": int(y2)},
            "center":     {"x": int(x1 + w // 2), "y": int(y1 + h // 2)},
            "size":       {"width": int(w), "height": int(h)},
            "confidence": round(float(all_scores[k]), 3),
            "source":     all_srcs[k],
            "label":      all_labels[k],
        })

    buttons.sort(key=lambda b: b["confidence"], reverse=True)
    # Re-index after sort
    for i, b in enumerate(buttons):
        b["id"] = i

    return buttons


def detect_buttons_to_json(img: np.ndarray,
                           img_path: str | None = None,
                           ) -> str:
    """Same as :func:`detect_buttons` but returns a pretty-printed JSON string."""
    buttons = detect_buttons(img, img_path)
    return json.dumps({
        "total_buttons": len(buttons),
        "buttons":       buttons,
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Annotated image helper
# ═══════════════════════════════════════════════════════════════════════════════

_COLOURS = {
    "yolo":    (0,   255, 0),    # green
    "ocr":     (0,   200, 255),  # orange
    "accent":  (255, 100, 100),  # light blue
    "contour": (255, 255, 0),    # cyan
}


def annotate_buttons(img: np.ndarray,
                     buttons: list[dict]) -> np.ndarray:
    """Draw detected buttons on a copy of *img* and return it."""
    out = img.copy()
    for btn in buttons:
        b = btn["bbox"]
        src = btn["source"]
        colour = _COLOURS.get(src, (200, 200, 200))
        thick = 2 if btn["confidence"] >= 0.70 else 1
        cv2.rectangle(out, (b["x1"], b["y1"]), (b["x2"], b["y2"]),
                      colour, thick)
        tag = btn["label"] or btn["source"]
        tag = f'{btn["id"]}:{tag} {btn["confidence"]:.0%}'
        ty = b["y1"] - 5 if b["y1"] > 15 else b["y2"] + 14
        cv2.putText(out, tag, (b["x1"], ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    img_path = str(Path(__file__).resolve().parent.parent / "image.png")
    img = cv2.imread(img_path)

    buttons = detect_buttons(img, img_path)
    print(detect_buttons_to_json(img, img_path))

    annotated = annotate_buttons(img, buttons)
    out = str(Path(__file__).resolve().parent.parent / "detected_buttons.png")
    cv2.imwrite(out, annotated)
    print(f"\n[✓] {len(buttons)} buttons → {out}")
