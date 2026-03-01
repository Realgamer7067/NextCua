"""
data_processing.py - Process recorded sessions into enriched dataset.

For each session in .sessions/:
  1. Read events.json  (repaired if truncated mid-write)
  2. Open recording.mp4 ONCE, extract frames at each event timestamp
  3. Run OCR + object-detection in parallel per frame  (asyncio + ThreadPool)
  4. Save enriched dataset.json + frames/ into .data/<session>/
  5. Delete the source session folder on success.

Usage:
    python -m dataset.data_processing          # from project root
    python dataset/data_processing.py          # direct
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_processing")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = _ROOT / ".sessions"
DATA_DIR = _ROOT / ".data"

# Frame encoding - JPEG is ~10x faster to encode than PNG and fine for ML data.
FRAME_EXT = ".jpg"
FRAME_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 90]

# ---------------------------------------------------------------------------
# Keyboard event grouping
# ---------------------------------------------------------------------------

# Modifier keys — never cause a screen change on their own.
_MODIFIER_KEYS = {
    "KEY_LEFTCTRL", "KEY_RIGHTCTRL",
    "KEY_LEFTALT", "KEY_RIGHTALT",
    "KEY_LEFTSHIFT", "KEY_RIGHTSHIFT",
    "KEY_LEFTMETA", "KEY_RIGHTMETA",
    "KEY_CAPSLOCK",
}

# Clean short names for display in hotkey combos.
_MOD_SHORT = {
    "KEY_LEFTCTRL": "Ctrl", "KEY_RIGHTCTRL": "Ctrl",
    "KEY_LEFTALT": "Alt", "KEY_RIGHTALT": "Alt",
    "KEY_LEFTSHIFT": "Shift", "KEY_RIGHTSHIFT": "Shift",
    "KEY_LEFTMETA": "Super", "KEY_RIGHTMETA": "Super",
    "KEY_CAPSLOCK": "CapsLock",
}

# Keys that produce a character when typed (batch-able into "type_text").
_CHAR_KEYS = {
    # Letters
    *(f"KEY_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    # Digits
    *(f"KEY_{d}" for d in "0123456789"),
    # Symbols / punctuation
    "KEY_MINUS", "KEY_EQUAL", "KEY_LEFTBRACE", "KEY_RIGHTBRACE",
    "KEY_SEMICOLON", "KEY_APOSTROPHE", "KEY_GRAVE", "KEY_BACKSLASH",
    "KEY_COMMA", "KEY_DOT", "KEY_SLASH", "KEY_SPACE",
    # Numpad
    "KEY_KPASTERISK", "KEY_KPMINUS", "KEY_KPPLUS", "KEY_KPDOT",
    *(f"KEY_KP{d}" for d in "0123456789"),
}

# Max gap (seconds) between consecutive character keys to batch them together.
_TEXT_BATCH_GAP = 0.35


def _group_keyboard_events(events: list[dict]) -> list[dict]:
    """
    Collapse raw keyboard events into smarter grouped events.

    Output event types:
      - type_text:  consecutive character keys → single event with "keys" list
                    (one frame for the whole burst)
      - hotkey:     modifier(s) + key combo → single event with "combo" string
                    (e.g. "Ctrl+C", "Alt+Tab", "Super")
      - press:      isolated non-character, non-modifier keys
                    (Enter, Escape, Backspace, F-keys, arrows, etc.)

    Non-keyboard events pass through untouched.
    """
    grouped: list[dict] = []
    text_buf: list[dict] = []          # buffered consecutive char-key events
    held_mods: list[str] = []          # modifier keys seen since last non-mod

    def _flush_text():
        """Emit one type_text event from the text buffer."""
        if not text_buf:
            return
        keys = [e["key_code"] for e in text_buf]
        grouped.append({
            "type": "keyboard",
            "action": "type_text",
            "keys": keys,
            "timestamp_ns": text_buf[-1].get("timestamp_ns", 0),
            "timestamp_sec": text_buf[-1]["timestamp_sec"],  # frame = end of burst
        })
        text_buf.clear()

    def _flush_mods_as_hotkey():
        """Emit held modifiers as a standalone hotkey (e.g. tapping Super)."""
        if not held_mods:
            return
        combo = "+".join(dict.fromkeys(held_mods))  # dedupe, preserve order
        grouped.append({
            "type": "keyboard",
            "action": "hotkey",
            "combo": combo,
            # Use the timestamp of the last modifier seen
            "timestamp_ns": _last_mod_event.get("timestamp_ns", 0),
            "timestamp_sec": _last_mod_event["timestamp_sec"],
        })
        held_mods.clear()

    _last_mod_event: dict = {}

    for ev in events:
        # Non-keyboard events break any active keyboard grouping
        if ev.get("type") != "keyboard":
            _flush_text()
            if held_mods:
                _flush_mods_as_hotkey()
            grouped.append(ev)
            continue

        kc = ev.get("key_code", "")

        # --- Modifier key ---
        if kc in _MODIFIER_KEYS:
            # If we had a text buffer going, flush it first
            _flush_text()
            short = _MOD_SHORT.get(kc, kc)
            if short not in held_mods:
                held_mods.append(short)
            _last_mod_event = ev
            continue

        # --- Character key ---
        if kc in _CHAR_KEYS and not held_mods:
            # Check time gap from last char
            if text_buf:
                gap = ev["timestamp_sec"] - text_buf[-1]["timestamp_sec"]
                if gap > _TEXT_BATCH_GAP:
                    _flush_text()
            text_buf.append(ev)
            continue

        # --- Character key WITH modifiers held → hotkey (e.g. Ctrl+C) ---
        if held_mods:
            _flush_text()
            # Strip "KEY_" prefix for the main key
            main_key = kc.replace("KEY_", "") if kc.startswith("KEY_") else kc
            combo = "+".join(held_mods) + "+" + main_key
            grouped.append({
                "type": "keyboard",
                "action": "hotkey",
                "combo": combo,
                "timestamp_ns": ev.get("timestamp_ns", 0),
                "timestamp_sec": ev["timestamp_sec"],
            })
            held_mods.clear()
            continue

        # --- Any other key (Enter, Escape, Backspace, arrows, F-keys, etc.) ---
        _flush_text()
        grouped.append({
            "type": "keyboard",
            "action": "press",
            "key_code": kc,
            "timestamp_ns": ev.get("timestamp_ns", 0),
            "timestamp_sec": ev["timestamp_sec"],
        })

    # Flush anything remaining
    _flush_text()
    if held_mods:
        _flush_mods_as_hotkey()

    return grouped


# ---------------------------------------------------------------------------
# Lazy model singletons  (thread-safe, avoid GPU allocation at import time)
# ---------------------------------------------------------------------------
_ocr_engine = None
_cv_engine = None
_model_lock = threading.Lock()


def _get_ocr_engine():
    """Return the RapidOCR engine, creating it on first call (thread-safe)."""
    global _ocr_engine
    if _ocr_engine is None:
        with _model_lock:
            if _ocr_engine is None:          # double-checked locking
                from rapidocr import (
                    EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR,
                )
                log.info("Loading OCR engine ...")
                _ocr_engine = RapidOCR(
                    params={
                        "Det.engine_type": EngineType.PADDLE,
                        "Det.lang_type": LangDet.EN,
                        "Det.model_type": ModelType.MOBILE,
                        "Det.ocr_version": OCRVersion.PPOCRV4,
                        "Rec.engine_type": EngineType.PADDLE,
                        "Rec.lang_type": LangRec.EN,
                        "Rec.model_type": ModelType.MOBILE,
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                        "EngineConfig.paddle.use_cuda": True,
                        "EngineConfig.paddle.cuda_ep_cfg.device_id": 0,
                    }
                )
    return _ocr_engine


def _get_cv_engine():
    """Return the RF-DETR detector, creating it on first call (thread-safe)."""
    global _cv_engine
    if _cv_engine is None:
        with _model_lock:
            if _cv_engine is None:            # double-checked locking
                from rfdetr.detr import RFDETRMedium
                weights = _ROOT / "dataset" / "models" / "Yolo" / "model.pth"
                log.info("Loading CV engine (RF-DETR) ...")
                model = RFDETRMedium(pretrain_weights=str(weights))
                if hasattr(model, "optimize_for_inference"):
                    try:
                        model.optimize_for_inference()
                        log.info("CV model optimised for inference.")
                    except Exception:
                        pass
                _cv_engine = model
    return _cv_engine


def warmup_models() -> None:
    """Load both models on the main thread so worker threads never race."""
    _get_ocr_engine()
    _get_cv_engine()


# ---------------------------------------------------------------------------
# Frame extraction - open video ONCE per session, yield one frame at a time
# ---------------------------------------------------------------------------

class VideoFrameExtractor:
    """Context manager that opens a video once and seeks to each timestamp."""

    def __init__(self, video_path: Path):
        self._path = video_path
        self._cap: cv2.VideoCapture | None = None
        self._total_frames = 0
        self._last_good_frame: np.ndarray | None = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            log.error("Could not open %s", self._path)
            self._cap = None
        else:
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return self

    def __exit__(self, *exc):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._last_good_frame = None

    def grab(self, timestamp_sec: float) -> np.ndarray | None:
        """Seek to *timestamp_sec* and return the frame (or None).

        For VFR videos (wf-recorder) where FRAME_COUNT is 0, falls back to
        the last successfully decoded frame so events recorded slightly after
        the video ends still get a frame.
        """
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ok, frame = self._cap.read()
        if not ok:
            # Try seeking to the last known frame index (works for CBR videos)
            if self._total_frames > 0:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, self._total_frames - 1))
                ok, frame = self._cap.read()
            # For VFR videos where FRAME_COUNT=0, reuse last successfully read frame
            if not ok and self._last_good_frame is not None:
                log.warning(
                    "Frame at %.3fs beyond video end — reusing last good frame",
                    timestamp_sec,
                )
                return self._last_good_frame
        if ok:
            self._last_good_frame = frame
        return frame if ok else None


# ---------------------------------------------------------------------------
# OCR / CV wrappers  (synchronous - dispatched to threads by the async layer)
# ---------------------------------------------------------------------------

def run_ocr(frame: np.ndarray) -> Any:
    """Run RapidOCR on a BGR frame."""
    try:
        return _get_ocr_engine()(frame, use_cls=False)
    except Exception as exc:
        return {"error": f"ocr: {exc}"}


def run_cv(frame: np.ndarray) -> Any:
    """Run RF-DETR object detection on a BGR frame (expects RGB)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        return _get_cv_engine().predict(rgb, threshold=0.7)
    except Exception as exc:
        return {"error": f"cv: {exc}"}


# ---------------------------------------------------------------------------
# Serialisation helper  (used internally by the cleaners)
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert an arbitrary object tree to JSON-safe primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    for method_name in ("to_dict", "as_dict"):
        fn = getattr(obj, method_name, None)
        if callable(fn):
            try:
                return _make_serializable(fn())
            except Exception:
                pass
    if hasattr(obj, "to_json") and callable(obj.to_json):
        try:
            return _make_serializable(json.loads(obj.to_json()))
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _make_serializable(vars(obj))
        except Exception:
            pass
    return str(obj)


# ---------------------------------------------------------------------------
# Result cleaners — extract only what's useful for training
# ---------------------------------------------------------------------------

def _clean_ocr(raw: Any) -> list[dict]:
    """
    Extract clean OCR results: list of {text, confidence, bbox}.

    Raw RapidOCR returns img (entire pixel array!), viser (engine internals),
    elapse (timing), etc. — none of that is useful for a training dataset.
    We keep only the text detections with their bounding boxes.
    """
    if isinstance(raw, dict) and "error" in raw:
        return []

    raw = _make_serializable(raw)
    if not isinstance(raw, dict):
        return []

    boxes = raw.get("boxes", [])
    texts = raw.get("txts", [])
    scores = raw.get("scores", [])

    results = []
    for box, text, score in zip(boxes, texts, scores):
        if not text or not text.strip():
            continue
        # Convert 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # to a simple axis-aligned bounding box [x_min, y_min, x_max, y_max]
        if isinstance(box, (list, tuple)) and len(box) == 4:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bbox = [round(min(xs), 1), round(min(ys), 1),
                    round(max(xs), 1), round(max(ys), 1)]
        else:
            bbox = box

        results.append({
            "text": text.strip(),
            "confidence": round(float(score), 4),
            "bbox": bbox,
        })

    return results


def _clean_cv(raw: Any) -> list[dict]:
    """
    Extract clean CV detections: list of {class_id, confidence, bbox}.

    Raw RF-DETR returns mask (None), tracker_id (None), data ({}) and
    xyxy with 15-decimal floats — all waste.  We keep only real detections.
    """
    if isinstance(raw, dict) and "error" in raw:
        return []

    raw = _make_serializable(raw)
    if not isinstance(raw, dict):
        return []

    xyxy = raw.get("xyxy", [])
    confs = raw.get("confidence", [])
    class_ids = raw.get("class_id", [])

    results = []
    for box, conf, cls in zip(xyxy, confs, class_ids):
        results.append({
            "class_id": int(cls),
            "confidence": round(float(conf), 4),
            "bbox": [round(float(c), 1) for c in box[:4]],
        })

    return results


# ---------------------------------------------------------------------------
# Click target resolution — match (x, y) to OCR/CV bounding boxes
# ---------------------------------------------------------------------------

def _point_in_bbox(x: float, y: float, bbox: list) -> bool:
    """Check if point (x, y) falls inside [x1, y1, x2, y2]."""
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _bbox_area(bbox: list) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _resolve_click_target(
    x: float, y: float,
    ocr: list[dict], cv: list[dict],
) -> dict | None:
    """
    Find what the user clicked on by hit-testing (x, y) against OCR text
    regions and CV-detected UI elements.

    Returns the best match (smallest bounding box containing the point),
    or None if the click didn't land on any detected element.

    Priority: if both an OCR text and a CV element contain the click,
    we pick the smallest one (most specific).
    """
    candidates: list[dict] = []

    for item in ocr:
        bbox = item.get("bbox")
        if bbox and _point_in_bbox(x, y, bbox):
            candidates.append({
                "source": "ocr",
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0),
                "bbox": bbox,
            })

    for item in cv:
        bbox = item.get("bbox")
        if bbox and _point_in_bbox(x, y, bbox):
            candidates.append({
                "source": "cv",
                "class_id": item.get("class_id"),
                "confidence": item.get("confidence", 0),
                "bbox": bbox,
            })

    if not candidates:
        return None

    # Pick the smallest bounding box — most specific element at that point.
    candidates.sort(key=lambda c: _bbox_area(c["bbox"]))
    return candidates[0]


# ---------------------------------------------------------------------------
# Core processing  (streaming, memory-friendly)
# ---------------------------------------------------------------------------

def _process_single_frame(
    frame: np.ndarray,
    frame_dir: Path,
    event_idx: int,
) -> tuple[str, list[dict], list[dict]]:
    """
    Save one frame to disk, run OCR then CV **sequentially** (keeps peak
    GPU memory to one model's activations at a time), return clean results.
    """
    fname = f"frame_{event_idx:05d}{FRAME_EXT}"
    cv2.imwrite(str(frame_dir / fname), frame, FRAME_ENCODE_PARAMS)

    ocr_data = _clean_ocr(run_ocr(frame))
    cv_data  = _clean_cv(run_cv(frame))

    return fname, ocr_data, cv_data


async def process_session(
    session_dir: Path,
    executor: ThreadPoolExecutor,
) -> bool:
    """
    Process one session → .data/<session>/dataset.json.

    Streams one event at a time: extract frame → OCR+CV → write → free.
    This keeps peak RAM to ~1 frame + 1 set of model activations.
    """
    name = session_dir.name
    events_path   = session_dir / "events.json"
    metadata_path = session_dir / "metadata.json"
    video_path    = session_dir / "recording.mp4"

    if not events_path.exists() or not video_path.exists():
        log.warning("Skipping %s - missing events.json or recording.mp4", name)
        return False

    # -- Load and repair events JSON (recorder may have been killed mid-write) --
    raw = events_path.read_text().rstrip()
    if not raw.endswith("]"):
        raw = raw.rstrip(",") + "\n]"
    try:
        events: list[dict] = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Skipping %s - corrupt events.json: %s", name, exc)
        return False

    if not events:
        log.warning("Skipping %s - 0 events", name)
        return False

    # -- Group consecutive keyboard events into type_text / hotkey / press --
    raw_count = len(events)
    events = _group_keyboard_events(events)
    if len(events) < raw_count:
        log.info(
            "%s - grouped %d raw events → %d smart events",
            name, raw_count, len(events),
        )

    metadata: dict = {}
    if metadata_path.exists():
        full_meta = json.loads(metadata_path.read_text())
        # Keep only training-relevant metadata
        metadata = {
            "session": name,
            "screen": full_meta.get("screen", {}),
            "duration_sec": full_meta.get("duration_sec"),
            "total_events": full_meta.get("total_events"),
        }

    # -- Output dirs --
    out_dir   = DATA_DIR / name
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "dataset.json"
    loop = asyncio.get_running_loop()
    total = len(events)
    t0 = time.perf_counter()

    # -- Stream enriched events to disk one-by-one --
    with open(out_path, "w") as fp, VideoFrameExtractor(video_path) as vfe:
        # Write JSON opening
        fp.write('{\n  "metadata": ')
        json.dump(metadata, fp, indent=2)
        fp.write(',\n  "events": [\n')

        for idx, event in enumerate(events):
            # Extract one frame
            frame = vfe.grab(event["timestamp_sec"])

            # Build a clean event — drop timestamp_ns (redundant with _sec)
            clean_event = {k: v for k, v in event.items() if k != "timestamp_ns"}

            if frame is None:
                enriched = {
                    **clean_event,
                    "frame": None,
                    "ocr": [],
                    "cv": [],
                }
            else:
                # Offload heavy OCR+CV to a worker thread so the event loop
                # stays responsive (the GIL is released inside the C backends).
                fname, ocr, cv = await loop.run_in_executor(
                    executor,
                    _process_single_frame,
                    frame, frame_dir, idx,
                )
                enriched = {**clean_event, "frame": fname, "ocr": ocr, "cv": cv}

            # For click events, resolve what was clicked on
            if (
                enriched.get("action") == "click"
                and "x" in enriched and "y" in enriched
                and (enriched.get("ocr") or enriched.get("cv"))
            ):
                target = _resolve_click_target(
                    enriched["x"], enriched["y"],
                    enriched.get("ocr", []), enriched.get("cv", []),
                )
                enriched["clicked_on"] = target  # None if click missed all boxes

            # Write this event immediately
            prefix = "    " if idx == 0 else ",\n    "
            fp.write(prefix + json.dumps(enriched))
            fp.flush()

            # Free the frame array right away
            del frame, enriched
            gc.collect()

            log.info(
                "%s - [%d/%d] %s %s @ %.3fs",
                name, idx + 1, total,
                event["type"], event.get("action", ""),
                event["timestamp_sec"],
            )

        # Close the JSON
        fp.write('\n  ]\n}\n')

    elapsed = time.perf_counter() - t0
    log.info(
        "%s - done: %d events in %.1fs (%.2f ev/s) -> %s",
        name, total, elapsed, total / max(elapsed, 0.001), out_path,
    )
    return True


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

async def process_all(*, delete_after: bool = True) -> None:
    """Discover sessions, process each, optionally delete source folder."""
    if not SESSIONS_DIR.exists():
        log.info("No .sessions directory - nothing to do.")
        return

    sessions = sorted(
        d for d in SESSIONS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    )
    if not sessions:
        log.info("No sessions to process.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Found %d session(s) to process.", len(sessions))

    # Warm up models on the main thread BEFORE spawning workers.
    warmup_models()

    # Single worker: OCR and CV run sequentially per frame to cap peak memory.
    executor = ThreadPoolExecutor(max_workers=1)

    for session_dir in sessions:
        log.info("=== Processing: %s ===", session_dir.name)
        try:
            ok = await process_session(session_dir, executor)
        except Exception:
            log.exception("Failed to process %s", session_dir.name)
            ok = False

        if ok and delete_after:
            shutil.rmtree(session_dir)
            log.info("Deleted source session: %s", session_dir.name)

    executor.shutdown(wait=False)
    log.info("All done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(process_all())
