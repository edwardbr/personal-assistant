#!/usr/bin/env python3
"""Right-Ctrl push-to-talk dictation: whisper-server -> ydotool.

Listens for KEY_RIGHTCTRL on every keyboard-capable input device, records
audio while the key is held, posts WAV to a local whisper-server OpenAI-shape
endpoint, and types the transcript via ydotool.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import wave
from typing import List

import evdev
import numpy as np
import requests
import sounddevice as sd


LOG = logging.getLogger("dictation")

SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE * 30 // 1000  # 30ms frames
MIN_AUDIO_SEC = 0.3
MAX_AUDIO_SEC = 60.0
KEY_PTT = evdev.ecodes.KEY_RIGHTCTRL
WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8771/v1/audio/transcriptions")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "en")
TYPE_DELAY_MS = int(os.environ.get("TYPE_DELAY_MS", "5"))  # ydotool inter-key delay


def find_keyboards() -> List[evdev.InputDevice]:
    """Open every input device whose capabilities look like a keyboard."""
    devices = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except (OSError, PermissionError) as e:
            LOG.debug("skip %s: %s", path, e)
            continue
        keys = dev.capabilities().get(evdev.ecodes.EV_KEY, [])
        if evdev.ecodes.KEY_A in keys and evdev.ecodes.KEY_Z in keys and KEY_PTT in keys:
            devices.append(dev)
            LOG.info("watching keyboard: %s (%s)", dev.name, path)
    if not devices:
        LOG.error("no keyboards found. is the user in the 'input' group? (`id` should list it)")
    return devices


class Recorder:
    """Background mic capture between start() and stop(). Returns int16 mono PCM bytes."""

    def __init__(self):
        self._stream = None
        self._frames: list[bytes] = []
        self._stop_evt = threading.Event()
        self._thread = None

    def start(self):
        if self._stream is not None:
            return
        self._frames = []
        self._stop_evt.clear()
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        max_frames = int(MAX_AUDIO_SEC * 1000 / 30)
        while not self._stop_evt.is_set():
            try:
                data, _ = self._stream.read(FRAME_SAMPLES)
            except Exception:
                break
            self._frames.append(bytes(data))
            if len(self._frames) > max_frames:
                LOG.warning("hit %.0fs cap; cutting recording", MAX_AUDIO_SEC)
                break

    def stop(self) -> bytes:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        return b"".join(self._frames)


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(pcm: bytes) -> str:
    wav = pcm_to_wav(pcm, SAMPLE_RATE)
    r = requests.post(
        WHISPER_URL,
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={
            "model": "whisper-1",
            "language": WHISPER_LANG,
            "response_format": "json",
        },
        timeout=60,
    )
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


TYPE_BACKEND = os.environ.get("DICTATION_TYPER", "auto").lower()  # auto | wtype | ydotool | clipboard
ENABLE_BEEPS = os.environ.get("DICTATION_BEEPS", "1") not in ("0", "false", "no", "off")
BEEP_VOLUME = float(os.environ.get("DICTATION_BEEP_VOLUME", "0.15"))


def play_tone(freq_hz: float, ms: int = 60):
    """Short non-blocking sine pip with attack/decay envelope (no clicks)."""
    if not ENABLE_BEEPS:
        return
    try:
        rate = 22050
        n = int(rate * ms / 1000)
        t = np.linspace(0, ms / 1000.0, n, endpoint=False, dtype=np.float32)
        env = np.ones(n, dtype=np.float32)
        a = min(n // 8, 200)
        d = min(n // 4, 800)
        if a > 0:
            env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if d > 0:
            env[-d:] = np.linspace(1.0, 0.0, d, dtype=np.float32)
        tone = (BEEP_VOLUME * env * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
        sd.play(tone, samplerate=rate, blocking=False)
    except Exception as e:
        LOG.debug("tone play failed: %s", e)


def _try_wtype(text: str) -> bool:
    """Wayland virtual-keyboard protocol; respects the active keymap (UK, US, etc)."""
    if not shutil.which("wtype"):
        return False
    try:
        subprocess.run(["wtype", "--", text], check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        LOG.warning("wtype failed: %s", e)
        return False


def _try_ydotool(text: str) -> bool:
    """uinput-level keystrokes; assumes US layout, so wrong on UK keyboards for @ \" # \\ etc.
    Useful as a fallback or for explicit ASCII-only contexts.
    """
    if not shutil.which("ydotool"):
        return False
    try:
        subprocess.run(["ydotool", "type", f"--key-delay={TYPE_DELAY_MS}", "--", text],
                       check=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        LOG.warning("ydotool failed: %s", e)
        return False


def _try_clipboard(text: str) -> bool:
    """Copy to clipboard and synthesize a paste keystroke. Layout-agnostic.
    Uses Ctrl+Shift+V (terminal-style) first since it works in both terminals and
    most GTK/Qt apps; falls back to Ctrl+V if Shift+Insert / Ctrl+Shift+V fails.
    """
    if not shutil.which("wl-copy"):
        LOG.error("wl-copy not installed; cannot use clipboard backend (sudo rpm-ostree install wl-clipboard)")
        return False
    try:
        subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True, timeout=5)
    except Exception as e:
        LOG.error("wl-copy failed: %s", e)
        return False
    # Try ydotool key for Ctrl+Shift+V (29=LeftCtrl, 42=LeftShift, 47=v) — keycodes are layout-independent
    if shutil.which("ydotool"):
        try:
            subprocess.run(["ydotool", "key", "29:1", "42:1", "47:1", "47:0", "42:0", "29:0"],
                           check=True, timeout=5)
            return True
        except Exception as e:
            LOG.warning("ydotool Ctrl+Shift+V failed: %s; transcript is still on clipboard for manual paste", e)
            return True  # text IS on clipboard; user can paste manually
    LOG.info("transcript copied to clipboard; press Ctrl+V (or Ctrl+Shift+V in terminals) to paste")
    return True


def type_text(text: str):
    if not text:
        return
    backend = TYPE_BACKEND
    if backend == "auto":
        # Prefer wtype (layout-aware) on wlroots-based compositors. GNOME/KDE do not
        # implement zwp_virtual_keyboard_v1, so wtype always fails there — skip straight
        # to clipboard. ydotool last (assumes US layout).
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
        wtype_supported = not any(d in desktop for d in ("GNOME", "KDE"))
        if os.environ.get("WAYLAND_DISPLAY") and wtype_supported and shutil.which("wtype"):
            backend = "wtype"
        elif shutil.which("wl-copy"):
            backend = "clipboard"
        elif shutil.which("ydotool"):
            backend = "ydotool"
        else:
            LOG.error("no typing backend available; install wtype OR ydotool OR wl-clipboard")
            return

    handlers = {"wtype": _try_wtype, "ydotool": _try_ydotool, "clipboard": _try_clipboard}
    handler = handlers.get(backend)
    if handler is None:
        LOG.error("unknown DICTATION_TYPER=%r (use: auto|wtype|ydotool|clipboard)", backend)
        return
    if handler(text):
        return
    # Auto fallback chain on failure
    if backend != "clipboard":
        LOG.info("falling back to clipboard")
        _try_clipboard(text)


def main():
    logging.basicConfig(
        level=os.environ.get("DICTATION_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--key",
        default=os.environ.get("DICTATION_KEY", "KEY_RIGHTCTRL"),
        help="evdev key name to use as PTT (default $DICTATION_KEY or KEY_RIGHTCTRL). "
             "Run `python -c 'import evdev; print([n for n in dir(evdev.ecodes) if n.startswith(\"KEY_\")])'` for the full list.",
    )
    args = ap.parse_args()
    global KEY_PTT
    try:
        KEY_PTT = getattr(evdev.ecodes, args.key)
    except AttributeError:
        LOG.error("unknown evdev key %r; check spelling (e.g. KEY_RIGHTCTRL, KEY_RIGHTSHIFT, KEY_SCROLLLOCK, KEY_F12)", args.key)
        return 2
    LOG.info("dictation daemon starting; PTT=%s; whisper=%s", args.key, WHISPER_URL)

    keyboards = find_keyboards()
    if not keyboards:
        return 1

    rec = Recorder()
    pressed = False
    press_t0 = 0.0

    fd_to_dev = {dev.fd: dev for dev in keyboards}
    while True:
        r, _, _ = select.select(list(fd_to_dev), [], [])
        for fd in r:
            dev = fd_to_dev[fd]
            try:
                events = list(dev.read())
            except OSError as e:
                LOG.warning("read err on %s: %s", dev.path, e)
                continue
            for ev in events:
                if ev.type != evdev.ecodes.EV_KEY or ev.code != KEY_PTT:
                    continue
                if ev.value == 1 and not pressed:
                    pressed = True
                    press_t0 = time.monotonic()
                    LOG.info("PTT down; recording...")
                    play_tone(880, 60)   # bright "open" tone
                    rec.start()
                elif ev.value == 0 and pressed:
                    pressed = False
                    held = time.monotonic() - press_t0
                    play_tone(660, 70)   # lower "close" tone
                    pcm = rec.stop()
                    if held < MIN_AUDIO_SEC:
                        LOG.info("(too short, %.2fs <= %.2fs, ignored)", held, MIN_AUDIO_SEC)
                        continue
                    LOG.info("PTT up after %.2fs; transcribing %d bytes...", held, len(pcm))
                    try:
                        text = transcribe(pcm)
                    except Exception as e:
                        LOG.warning("transcribe failed: %s", e)
                        continue
                    if not text:
                        LOG.info("(empty transcript)")
                        continue
                    LOG.info("typing: %s", text)
                    type_text(text)


if __name__ == "__main__":
    sys.exit(main() or 0)
