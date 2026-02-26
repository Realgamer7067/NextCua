import os
import sys
import json
import shutil
import cv2
from pathlib import Path
from rapidocr import RapidOCR,EngineType,LangDet,ModelType,OCRVersion,LangRec




SESSIONS_DIR = Path(__file__).parent.parent / ".sessions"
DATA_DIR = Path(__file__).parent.parent / ".data"
REC_MODEL = r"dataset/models/RapidOCR/en_PP-OCRv5_rec_mobile_infer.onnx"
DET_MODEL = r"dataset/models/RapidOCR/ch_PP-OCRv5_mobile_det.onnx"
engine = RapidOCR(
    params={
        "Det.engine_type": EngineType.PADDLE,
        "Det.lang_type": LangDet.EN,
        # "Det.model_path": DET_MODEL,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV4,
        "Rec.engine_type": EngineType.PADDLE,
        "Rec.lang_type": LangRec.EN,
        # "Rec.model_path": REC_MODEL,
        "Rec.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "EngineConfig.paddle.use_cuda": True,
        "EngineConfig.paddle.cuda_ep_cfg.device_id": 0
    }
)


def extract_frame(video_path, timestamp_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return None
    # try requested time (ms)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_sec) * 1000.0)
    ret, frame = cap.read()
    if not ret:
        # try fallback to last frame
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    return frame
        except Exception:
            pass
        cap.release()
        print(f"  ERROR: Could not read frame at {timestamp_sec:.3f}s")
        return None
    cap.release()
    return frame


def run_ocr(frame):
    try:
        results = engine(frame, use_cls=False)
    except Exception as e:
        return {"error": f"ocr_error: {e}"}
    return results


def run_cv(frame):
    # TODO: implement CV/object detection — return list of detected UI elements
    # expected format:
    # [
    #     {"label": "button", "bbox": [x1, y1, x2, y2], "confidence": 0.95},
    #     ...
    # ]
    return []


def _make_serializable(obj):
    # basic primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # rapidocr types may provide to_dict/to_json
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _make_serializable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        try:
            return _make_serializable(obj.as_dict())
        except Exception:
            pass
    if hasattr(obj, "to_json") and callable(obj.to_json):
        try:
            import json as _json

            return _make_serializable(_json.loads(obj.to_json()))
        except Exception:
            pass
    # fallback to __dict__ if available
    if hasattr(obj, "__dict__"):
        try:
            return _make_serializable(vars(obj))
        except Exception:
            pass
    # last resort: string representation
    try:
        return str(obj)
    except Exception:
        return {"error": "unserializable_object"}


def process_event(event, video_path, frame_dir, event_idx):
    timestamp_sec = event["timestamp_sec"]
    frame = extract_frame(video_path, timestamp_sec)
    if frame is None:
        # record the event with an error flag instead of skipping entirely
        return {**event, "frame": None, "ocr": {"error": "no_frame"}, "cv": {"error": "no_frame"}}

    frame_filename = f"frame_{event_idx:05d}.png"
    frame_path = frame_dir / frame_filename
    cv2.imwrite(str(frame_path), frame)

    try:
        ocr_data = run_ocr(frame)
    except Exception as e:
        ocr_data = {"error": f"ocr_exception: {e}"}
    try:
        cv_data = run_cv(frame)
    except Exception as e:
        cv_data = {"error": f"cv_exception: {e}"}

    # ensure serializable
    ocr_data = _make_serializable(ocr_data)
    cv_data = _make_serializable(cv_data)

    enriched = {
        **event,
        "frame": frame_filename,
        "ocr": ocr_data,
        "cv": cv_data,
    }
    return enriched


def process_session(session_dir):
    session_name = session_dir.name
    events_path = session_dir / "events.json"
    metadata_path = session_dir / "metadata.json"
    video_path = session_dir / "recording.mp4"

    if not events_path.exists() or not video_path.exists():
        print(f"  Skipping {session_name}: missing events.json or recording.mp4")
        return False

    with open(events_path, "r") as f:
        raw = f.read().rstrip()
        if not raw.endswith("]"):
            if raw.endswith(","):
                raw = raw[:-1]
            raw += "\n]"
        events = json.loads(raw)

    if not events:
        print(f"  Skipping {session_name}: no events")
        return False

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    output_dir = DATA_DIR / session_name
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    enriched_events = []
    for idx, event in enumerate(events):
        print(f"  Processing event {idx + 1}/{len(events)}: "
              f"{event['type']} {event.get('action', '')} @ {event['timestamp_sec']:.3f}s")
        result = process_event(event, video_path, frame_dir, idx)
        if result is not None:
            enriched_events.append(result)

    output = {
        "metadata": metadata,
        "events": enriched_events,
    }

    output_path = output_dir / "dataset.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(enriched_events)} events → {output_path}")
    return True


def process_all(delete_after=True):
    if not SESSIONS_DIR.exists():
        print("No .sessions directory found.")
        return

    sessions = sorted([
        d for d in SESSIONS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("session_")
    ])

    if not sessions:
        print("No sessions to process.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(sessions)} session(s) to process.\n")

    for session_dir in sessions:
        print(f"Processing: {session_dir.name}")
        success = process_session(session_dir)
        if success and delete_after:
            shutil.rmtree(session_dir)
            print(f"  Deleted session: {session_dir.name}")
        print()

    print("Done.")


if __name__ == "__main__":
    keep = "--keep" in sys.argv
    process_all(delete_after=not keep)