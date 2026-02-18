import os
import sys
import json
import shutil
import cv2
from pathlib import Path

SESSIONS_DIR = Path(__file__).parent.parent / ".sessions"
DATA_DIR = Path(__file__).parent.parent / ".data"


def extract_frame(video_path, timestamp_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  ERROR: Could not read frame at {timestamp_sec:.3f}s")
        return None
    return frame


def run_ocr(frame):
    # TODO: implement OCR — return list of detected text regions
    # expected format:
    # [
    #     {"text": "File", "bbox": [x1, y1, x2, y2], "confidence": 0.98},
    #     ...
    # ]
    return []


def run_cv(frame):
    # TODO: implement CV/object detection — return list of detected UI elements
    # expected format:
    # [
    #     {"label": "button", "bbox": [x1, y1, x2, y2], "confidence": 0.95},
    #     ...
    # ]
    return []


def process_event(event, video_path, frame_dir, event_idx):
    timestamp_sec = event["timestamp_sec"]
    frame = extract_frame(video_path, timestamp_sec)
    if frame is None:
        return None

    frame_filename = f"frame_{event_idx:05d}.png"
    frame_path = frame_dir / frame_filename
    cv2.imwrite(str(frame_path), frame)

    ocr_data = run_ocr(frame)
    cv_data = run_cv(frame)

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