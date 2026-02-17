import asyncio
import subprocess
import signal
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path
from evdev import InputDevice, list_devices, ecodes, categorize, AbsInfo

SESSIONS_DIR = Path(__file__).parent.parent / ".sessions"

MOUSE_BTNS = {
    ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE,
    ecodes.BTN_SIDE, ecodes.BTN_EXTRA,
}

TOUCHPAD_FINGER_BTNS = {
    ecodes.BTN_TOOL_FINGER,
    ecodes.BTN_TOOL_DOUBLETAP,
    ecodes.BTN_TOOL_TRIPLETAP,
    ecodes.BTN_TOOL_QUADTAP,
    ecodes.BTN_TOOL_QUINTTAP,
}


class InputRecorder:
    def __init__(self):
        self.event_count = 0
        self.start_time_ns = 0
        self.is_recording = False
        self.recorder_process = None
        self.session_dir = None
        self.video_path = None
        self.events_path = None
        self.metadata_path = None
        self._loop = None
        self._events_file = None
        self._last_timestamp_ns = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.screen_w = 0
        self.screen_h = 0

    def _detect_screen_size(self):
        try:
            out = subprocess.check_output(
                ["wlr-randr"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                line = line.strip()
                if "current" in line.lower():
                    parts = line.split()
                    for p in parts:
                        if "x" in p and p[0].isdigit():
                            w, h = p.split("x")
                            self.screen_w = int(w)
                            self.screen_h = int(h)
                            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        try:
            out = subprocess.check_output(
                ["swaymsg", "-t", "get_outputs", "-r"],
                text=True, stderr=subprocess.DEVNULL
            )
            outputs = json.loads(out)
            for o in outputs:
                if o.get("active"):
                    rect = o.get("rect", {})
                    self.screen_w = rect.get("width", 1920)
                    self.screen_h = rect.get("height", 1080)
                    return
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        self.screen_w = 1920
        self.screen_h = 1080

    def _create_session_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = SESSIONS_DIR / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.session_dir / "recording.mp4"
        self.events_path = self.session_dir / "events.json"
        self.metadata_path = self.session_dir / "metadata.json"
        print(f"Session directory: {self.session_dir}")

    def _open_events_file(self):
        self._events_file = open(self.events_path, "w")
        self._events_file.write("[\n")
        self._events_file.flush()

    def _write_event(self, entry):
        if self._events_file is None:
            return
        prefix = "  " if self.event_count == 0 else ",\n  "
        self._events_file.write(prefix + json.dumps(entry))
        self._events_file.flush()
        self.event_count += 1
        self._last_timestamp_ns = entry["timestamp_ns"]

    def _close_events_file(self):
        if self._events_file is None:
            return
        self._events_file.write("\n]\n")
        self._events_file.close()
        self._events_file = None

    @staticmethod
    def repair_events_file(path):
        path = Path(path)
        if not path.exists():
            return
        raw = path.read_text().rstrip()
        if raw.endswith("]"):
            return
        if raw.endswith(","):
            raw = raw[:-1]
        raw += "\n]\n"
        path.write_text(raw)

    def find_devices(self):
        devices = [InputDevice(path) for path in list_devices()]
        keyboards = []
        mice = []
        touchpads = []

        print("Scanning input devices...")
        for dev in devices:
            caps = dev.capabilities()
            key_codes = set(caps.get(ecodes.EV_KEY, []))
            has_real_keys = bool(key_codes & {ecodes.KEY_A, ecodes.KEY_Z, ecodes.KEY_SPACE})
            has_mouse_btn = bool(key_codes & MOUSE_BTNS)
            has_touchpad_btn = bool(key_codes & TOUCHPAD_FINGER_BTNS)
            has_rel = ecodes.EV_REL in caps
            has_abs = ecodes.EV_ABS in caps

            if has_real_keys and not has_mouse_btn and not has_touchpad_btn:
                keyboards.append(dev)
                print(f"  [KEYBOARD]  {dev.path}: {dev.name}")
            elif has_touchpad_btn and has_abs:
                touchpads.append(dev)
                print(f"  [TOUCHPAD]  {dev.path}: {dev.name}")
            elif has_mouse_btn and has_rel and not has_touchpad_btn:
                mice.append(dev)
                print(f"  [MOUSE]     {dev.path}: {dev.name}")

        if not keyboards:
            print("ERROR: No keyboard found.")
        if not mice and not touchpads:
            print("ERROR: No pointer device found.")

        return keyboards, mice, touchpads

    async def record_keyboard(self, device):
        print(f"  Listening keyboard: {device.name} ({device.path})")
        try:
            async for event in device.async_read_loop():
                if not self.is_recording:
                    continue
                if event.type != ecodes.EV_KEY:
                    continue
                key_event = categorize(event)
                if key_event.keystate != 1:
                    continue
                relative_ns = time.time_ns() - self.start_time_ns
                entry = {
                    "type": "keyboard",
                    "action": "press",
                    "key_code": key_event.keycode,
                    "scancode": key_event.scancode,
                    "timestamp_ns": relative_ns,
                    "timestamp_sec": round(relative_ns / 1_000_000_000, 6),
                }
                self._write_event(entry)
                print(f"  [KEY] {key_event.keycode} @ {entry['timestamp_sec']:.3f}s")
        except OSError:
            print(f"  Keyboard {device.name} disconnected.")

    async def record_mouse(self, device):
        print(f"  Listening mouse: {device.name} ({device.path})")
        try:
            async for event in device.async_read_loop():
                if not self.is_recording:
                    continue
                if event.type == ecodes.EV_REL:
                    if event.code == ecodes.REL_X:
                        self.mouse_x += event.value
                    elif event.code == ecodes.REL_Y:
                        self.mouse_y += event.value
                    elif event.code in (ecodes.REL_WHEEL, ecodes.REL_HWHEEL):
                        direction = "vertical" if event.code == ecodes.REL_WHEEL else "horizontal"
                        relative_ns = time.time_ns() - self.start_time_ns
                        entry = {
                            "type": "mouse",
                            "action": "scroll",
                            "direction": direction,
                            "delta": event.value,
                            "x": self.mouse_x,
                            "y": self.mouse_y,
                            "timestamp_ns": relative_ns,
                            "timestamp_sec": round(relative_ns / 1_000_000_000, 6),
                        }
                        self._write_event(entry)
                        print(
                            f"  [SCROLL] {direction} delta={event.value} "
                            f"({self.mouse_x}, {self.mouse_y}) @ {entry['timestamp_sec']:.3f}s"
                        )
                elif event.type == ecodes.EV_KEY and event.code in MOUSE_BTNS:
                    key_event = categorize(event)
                    if key_event.keystate != 1:
                        continue
                    relative_ns = time.time_ns() - self.start_time_ns
                    entry = {
                        "type": "mouse",
                        "action": "click",
                        "button": key_event.keycode,
                        "x": self.mouse_x,
                        "y": self.mouse_y,
                        "timestamp_ns": relative_ns,
                        "timestamp_sec": round(relative_ns / 1_000_000_000, 6),
                    }
                    self._write_event(entry)
                    print(
                        f"  [CLICK] {key_event.keycode} ({self.mouse_x}, {self.mouse_y}) "
                        f"@ {entry['timestamp_sec']:.3f}s"
                    )
        except OSError:
            print(f"  Mouse {device.name} disconnected.")

    async def record_touchpad(self, device):
        print(f"  Listening touchpad: {device.name} ({device.path})")

        abs_caps = device.capabilities().get(ecodes.EV_ABS, [])
        abs_x_info = None
        abs_y_info = None
        for code, absinfo in abs_caps:
            if code == ecodes.ABS_X:
                abs_x_info = absinfo
            elif code == ecodes.ABS_Y:
                abs_y_info = absinfo

        raw_x = 0
        raw_y = 0
        finger_count = 0
        touching = False

        try:
            async for event in device.async_read_loop():
                if not self.is_recording:
                    continue

                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X:
                        raw_x = event.value
                    elif event.code == ecodes.ABS_Y:
                        raw_y = event.value

                elif event.type == ecodes.EV_KEY:
                    if event.code == ecodes.BTN_TOOL_FINGER:
                        if event.value == 1:
                            finger_count = max(finger_count, 1)
                        else:
                            finger_count = 0
                    elif event.code == ecodes.BTN_TOOL_DOUBLETAP:
                        if event.value == 1:
                            finger_count = 2
                    elif event.code == ecodes.BTN_TOOL_TRIPLETAP:
                        if event.value == 1:
                            finger_count = 3
                    elif event.code == ecodes.BTN_TOOL_QUADTAP:
                        if event.value == 1:
                            finger_count = 4
                    elif event.code == ecodes.BTN_TOOL_QUINTTAP:
                        if event.value == 1:
                            finger_count = 5
                    elif event.code == ecodes.BTN_LEFT and event.value == 1:
                        screen_x, screen_y = self._scale_touchpad(
                            raw_x, raw_y, abs_x_info, abs_y_info
                        )
                        if finger_count >= 2:
                            button = "BTN_RIGHT"
                        else:
                            button = "BTN_LEFT"
                        relative_ns = time.time_ns() - self.start_time_ns
                        entry = {
                            "type": "touchpad",
                            "action": "click",
                            "button": button,
                            "x": screen_x,
                            "y": screen_y,
                            "fingers": finger_count,
                            "timestamp_ns": relative_ns,
                            "timestamp_sec": round(relative_ns / 1_000_000_000, 6),
                        }
                        self._write_event(entry)
                        print(
                            f"  [TAP] {button} ({screen_x}, {screen_y}) "
                            f"fingers={finger_count} @ {entry['timestamp_sec']:.3f}s"
                        )
                    elif event.code == ecodes.BTN_TOUCH:
                        if event.value == 1:
                            touching = True
                        else:
                            touching = False

        except OSError:
            print(f"  Touchpad {device.name} disconnected.")

    def _scale_touchpad(self, raw_x, raw_y, abs_x_info, abs_y_info):
        if abs_x_info and abs_y_info and self.screen_w and self.screen_h:
            x = int((raw_x - abs_x_info.min) / (abs_x_info.max - abs_x_info.min) * self.screen_w)
            y = int((raw_y - abs_y_info.min) / (abs_y_info.max - abs_y_info.min) * self.screen_h)
            return max(0, min(x, self.screen_w)), max(0, min(y, self.screen_h))
        return raw_x, raw_y

    def _save_metadata(self, keyboards, mice, touchpads):
        metadata = {
            "session_start": datetime.now().isoformat(),
            "screen": {"width": self.screen_w, "height": self.screen_h},
            "keyboards": [{"name": d.name, "path": d.path} for d in keyboards],
            "mice": [{"name": d.name, "path": d.path} for d in mice],
            "touchpads": [{"name": d.name, "path": d.path} for d in touchpads],
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _setup_signal_handler(self):
        def _handle_signal():
            for task in asyncio.all_tasks(self._loop):
                task.cancel()
        self._loop.add_signal_handler(signal.SIGINT, _handle_signal)

    async def run(self):
        self._loop = asyncio.get_running_loop()
        self._setup_signal_handler()

        self._detect_screen_size()
        print(f"Screen: {self.screen_w}x{self.screen_h}")

        keyboards, mice, touchpads = self.find_devices()
        if not keyboards or (not mice and not touchpads):
            print("Missing devices. Check permissions (need root or input group).")
            return

        self._create_session_dir()
        self._save_metadata(keyboards, mice, touchpads)
        self._open_events_file()

        print(f"Starting fullscreen recording → {self.video_path}")
        try:
            self.recorder_process = subprocess.Popen(
                ["wf-recorder", "-f", str(self.video_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except FileNotFoundError:
            print("Error: 'wf-recorder' not found.")
            sys.exit(1)

        self.start_time_ns = time.time_ns()
        self.is_recording = True
        print("Recording! Press Ctrl+C to stop.\n")

        tasks = []
        for dev in keyboards:
            tasks.append(self.record_keyboard(dev))
        for dev in mice:
            tasks.append(self.record_mouse(dev))
        for dev in touchpads:
            tasks.append(self.record_touchpad(dev))

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        if not self.is_recording and not self.recorder_process:
            return

        print("\nStopping recording...")
        self.is_recording = False

        if self.recorder_process and self.recorder_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.recorder_process.pid), signal.SIGINT)
                self.recorder_process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                self.recorder_process.kill()
            self.recorder_process = None

        self._close_events_file()
        print(f"Saved {self.event_count} events → {self.events_path}")

        if self.metadata_path and self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["session_end"] = datetime.now().isoformat()
            metadata["total_events"] = self.event_count
            if self._last_timestamp_ns:
                metadata["duration_sec"] = round(
                    self._last_timestamp_ns / 1_000_000_000, 3
                )
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

        print(f"Session saved to: {self.session_dir}")
        print("Done.")


if __name__ == "__main__":
    recorder = InputRecorder()
    try:
        asyncio.run(recorder.run())
    except KeyboardInterrupt:
        recorder.stop()