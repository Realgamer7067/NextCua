# Dataset Pipeline

## Overview

This pipeline records user desktop sessions and processes them into structured
training data for a next-action prediction model.

```
record (recoder.py)          process (data_processing.py)
 ┌──────────────┐             ┌──────────────────┐
 │ .sessions/   │  ────────►  │ .data/           │
 │  session_*/  │             │  session_*/      │
 │   events.json│             │   dataset.json   │
 │   metadata   │             │   frames/        │
 │   recording  │             │    frame_00000   │
 └──────────────┘             └──────────────────┘
```

## Recording

```bash
sudo python -m dataset.recoder
```

Captures keyboard, mouse, touchpad events + screen video via `wf-recorder`
at 60 FPS CFR. Each event also records the **focused window** info
(class, title, position, size) via Hyprland IPC.
Press `Ctrl+C` to stop. Output goes to `.sessions/session_<timestamp>/`.

## Processing

```bash
python -m dataset.data_processing
```

For each session in `.sessions/`:
1. **Groups keyboard events** — consecutive typing → `type_text`, modifier combos → `hotkey`
2. Extracts a frame from the video at each grouped event timestamp
3. Runs **OCR** (RapidOCR) to detect all text on screen
4. Runs **object detection** (RF-DETR) to detect UI elements
5. **Resolves click targets** — matches click coordinates against OCR/CV bounding boxes
6. Writes a clean `dataset.json` + frame images to `.data/<session>/`
7. Deletes the source session on success

## Output Schema

### `dataset.json`

```jsonc
{
  "metadata": {
    "session": "session_2026-02-27_20-56-52",
    "screen": {"width": 1920, "height": 1080},
    "duration_sec": 9.5,
    "total_events": 12
  },
  "events": [
    // --- Typing burst (e.g. user typed "hello") ---
    {
      "type": "keyboard",
      "action": "type_text",
      "keys": ["KEY_H", "KEY_E", "KEY_L", "KEY_L", "KEY_O"],
      "timestamp_sec": 2.841,
      "window": {
        "class": "firefox",
        "title": "GitHub - Mozilla Firefox",
        "position": [6, 51],
        "size": [1908, 1023]
      },
      "frame": "frame_00000.jpg",
      "ocr": [...],
      "cv": [...]
    },

    // --- Hotkey combo (e.g. Ctrl+S) ---
    {
      "type": "keyboard",
      "action": "hotkey",
      "combo": "Ctrl+S",
      "timestamp_sec": 3.102,
      "window": { ... },
      "frame": "frame_00001.jpg",
      "ocr": [...],
      "cv": [...]
    },

    // --- Isolated special key (e.g. Enter) ---
    {
      "type": "keyboard",
      "action": "press",
      "key_code": "KEY_ENTER",
      "timestamp_sec": 3.500,
      "window": { ... },
      "frame": "frame_00002.jpg",
      "ocr": [...],
      "cv": [...]
    },

    // --- Mouse / touchpad click with target resolution ---
    {
      "type": "touchpad",
      "action": "click",
      "button": "BTN_LEFT",
      "x": 545, "y": 105,
      "timestamp_sec": 1.732,
      "window": {
        "class": "google-chrome",
        "title": "Google Gemini - Google Chrome",
        "position": [6, 51],
        "size": [1908, 1023]
      },
      "frame": "frame_00003.jpg",
      "ocr": [
        {
          "text": "File",
          "confidence": 0.9812,
          "bbox": [10.0, 5.0, 45.0, 22.0]
        }
      ],
      "cv": [
        {
          "class_id": 0,
          "confidence": 0.8261,
          "bbox": [296.4, 84.9, 409.9, 126.5]
        }
      ],
      "clicked_on": {
        "source": "ocr",
        "text": "File",
        "confidence": 0.9812,
        "bbox": [10.0, 5.0, 45.0, 22.0]
      }
    }
  ]
}
```

### Keyboard grouping

During processing, raw keyboard events are grouped:

| Action | When | Example |
|--------|------|---------|
| `type_text` | Consecutive character keys within 350ms gaps | Typing "hello world" → 1 event |
| `hotkey` | Modifier(s) + key combo | Ctrl+C, Alt+Tab, Super |
| `press` | Isolated special keys | Enter, Escape, Backspace, F1, arrows |

This dramatically reduces frames to process — a 50-keystroke typing burst
becomes 1 frame instead of 50.

### Window context

Every event includes a `window` field with the focused window at the time of
the event, queried from Hyprland via `hyprctl activewindow -j`:

| Field | Description |
|-------|-------------|
| `class` | Application class name (e.g. `"google-chrome"`, `"code"`) |
| `title` | Window title (e.g. `"main.py - Visual Studio Code"`) |
| `position` | `[x, y]` top-left position on screen |
| `size` | `[width, height]` of the window |

### Click target resolution

Click events (`action: "click"`) include a `clicked_on` field that identifies
what UI element (if any) was under the cursor. The system hit-tests the click
`(x, y)` against all OCR text regions and CV-detected bounding boxes, then
picks the **smallest** (most specific) match.

| Field | Description |
|-------|-------------|
| `source` | `"ocr"` or `"cv"` |
| `text` | OCR text content (only for `source: "ocr"`) |
| `class_id` | Detection class ID (only for `source: "cv"`) |
| `confidence` | Model confidence score |
| `bbox` | `[x1, y1, x2, y2]` bounding box of the matched element |

`clicked_on` is `null` when the click doesn't land on any detected element.

### Mouse coordinates

Mouse coordinates (`x`, `y`) are **absolute screen pixel positions**. The
recorder seeds the initial cursor position from Hyprland (`hyprctl cursorpos`)
at recording start, then accumulates relative deltas, clamped to screen bounds.

### Bounding boxes

All bounding boxes use **`[x1, y1, x2, y2]`** format (top-left, bottom-right)
in pixel coordinates matching the frame dimensions.

### Frame images

JPEG files at quality 90, named `frame_NNNNN.jpg` (zero-padded index matching
the event index in the events array).
