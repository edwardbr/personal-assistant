#!/usr/bin/env python3
"""Right-Ctrl push-to-talk dictation: whisper-server -> ydotool.

Listens for KEY_RIGHTCTRL on every keyboard-capable input device, records
audio while the key is held, posts WAV to a local whisper-server OpenAI-shape
endpoint, and types the transcript via ydotool.

Optional chord: if DICTATION_CHORD_KEY is set (default KEY_RIGHTSHIFT), holding
the PTT key and the chord key simultaneously runs a headless voice-assistant
path by default: Whisper transcript -> llama.cpp chat endpoint -> spoken TTS
answer. Set DICTATION_CHORD_ACTION=talon to restore the old Talon toggle.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional, Set

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
KEY_CHORD: Optional[int] = evdev.ecodes.KEY_RIGHTSHIFT  # set to None to disable chord
TALON_SOCK = Path(os.environ.get("TALON_CHORD_SOCKET", str(Path.home() / ".talon" / "chord.sock")))
WHISPER_URL = os.environ.get("WHISPER_URL", "http://127.0.0.1:8771/v1/audio/transcriptions")
WHISPER_LANG = os.environ.get("WHISPER_LANG", "en")
TYPE_DELAY_MS = int(os.environ.get("TYPE_DELAY_MS", "5"))  # ydotool inter-key delay

ASSISTANT_URL = os.environ.get("DICTATION_ASSISTANT_URL", "http://127.0.0.1:3001/v1/chat/completions")
ASSISTANT_MODEL = os.environ.get("DICTATION_ASSISTANT_MODEL", "auto")
ASSISTANT_API_KEY = os.environ.get("DICTATION_ASSISTANT_API_KEY", "")
ASSISTANT_SYSTEM_PROMPT = os.environ.get(
    "DICTATION_ASSISTANT_SYSTEM_PROMPT",
    "You are a local voice assistant. Answer directly and briefly. Do not include reasoning.",
)
ASSISTANT_MAX_TOKENS = int(os.environ.get("DICTATION_ASSISTANT_MAX_TOKENS", "256"))
ASSISTANT_TEMPERATURE = float(os.environ.get("DICTATION_ASSISTANT_TEMPERATURE", "0.2"))
ASSISTANT_DISABLE_THINKING = os.environ.get("DICTATION_ASSISTANT_DISABLE_THINKING", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
ASSISTANT_TTS_ENABLED = os.environ.get("DICTATION_ASSISTANT_TTS", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
TTS_VOICE_PATH = os.environ.get(
    "DICTATION_TTS_VOICE_PATH",
    str(Path.home() / "Models/tts/en/en_GB/alba/medium/en_GB-alba-medium.onnx"),
)
TTS_TOOLBOX = os.environ.get("DICTATION_TTS_TOOLBOX", "whisper-rocm-7.2.3")
TTS_ESPEAK_VOICE = os.environ.get("DICTATION_TTS_ESPEAK_VOICE", "en-gb")

_ASSISTANT_RESOLVED_MODEL: Optional[str] = None


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


def _assistant_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ASSISTANT_API_KEY}"} if ASSISTANT_API_KEY else {}


def _assistant_base_url() -> str:
    return ASSISTANT_URL.split("/v1/")[0].rstrip("/")


def resolve_assistant_model() -> str:
    global _ASSISTANT_RESOLVED_MODEL
    if ASSISTANT_MODEL != "auto":
        return ASSISTANT_MODEL
    try:
        r = requests.get(f"{_assistant_base_url()}/v1/models", headers=_assistant_headers(), timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        loaded = [m for m in models if (m.get("status", {}) or {}).get("value") == "loaded"]
        healthy = [m for m in models if not (m.get("status", {}) or {}).get("failed")]
        chosen = (loaded or healthy or models)[0]["id"] if (loaded or healthy or models) else "default"
        if chosen != _ASSISTANT_RESOLVED_MODEL:
            LOG.info("assistant model auto-resolved: %s", chosen)
            _ASSISTANT_RESOLVED_MODEL = chosen
        return chosen
    except Exception as e:
        LOG.warning("assistant model resolution failed: %s", e)
        return _ASSISTANT_RESOLVED_MODEL or "default"


def ask_assistant(text: str) -> str:
    payload = {
        "model": resolve_assistant_model(),
        "messages": [
            {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": ASSISTANT_MAX_TOKENS,
        "temperature": ASSISTANT_TEMPERATURE,
        "stream": False,
    }
    if ASSISTANT_DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    r = requests.post(ASSISTANT_URL, json=payload, headers=_assistant_headers(), timeout=120)
    r.raise_for_status()
    message = r.json()["choices"][0]["message"]
    answer = (message.get("content") or "").strip()
    if answer:
        return answer
    if message.get("reasoning_content"):
        LOG.warning("assistant returned reasoning but no final answer; not speaking reasoning text")
    return ""


def _piper_sample_rate(voice_path: str) -> int:
    try:
        with open(voice_path + ".json") as f:
            return int(json.load(f).get("audio", {}).get("sample_rate", 22050))
    except Exception as e:
        LOG.debug("could not read Piper sample rate from %s.json: %s", voice_path, e)
        return 22050


def _piper_cmd(voice_path: str) -> Optional[list[str]]:
    if shutil.which("piper"):
        return ["piper", "-m", voice_path, "--output-raw"]
    if shutil.which("podman"):
        return ["podman", "exec", "-i", TTS_TOOLBOX, "piper", "-m", voice_path, "--output-raw"]
    return None


def _speak_with_piper(text: str) -> bool:
    cmd = _piper_cmd(TTS_VOICE_PATH)
    if cmd is None:
        return False
    try:
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=45,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOG.warning("Piper TTS failed: %s", e)
        return False
    if not proc.stdout:
        LOG.warning("Piper TTS produced no audio")
        return False
    audio = np.frombuffer(proc.stdout, dtype=np.int16)
    try:
        sd.play(audio, samplerate=_piper_sample_rate(TTS_VOICE_PATH), blocking=True)
        return True
    except Exception as e:
        LOG.warning("TTS playback failed: %s", e)
        return False


def _speak_with_espeak(text: str) -> bool:
    if not shutil.which("espeak-ng"):
        return False
    try:
        subprocess.run(["espeak-ng", "-v", TTS_ESPEAK_VOICE, text], check=True, timeout=45)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        LOG.warning("espeak-ng TTS failed: %s", e)
        return False


def speak_text(text: str) -> None:
    text = text.strip()
    if not text or not ASSISTANT_TTS_ENABLED:
        return
    if _speak_with_piper(text):
        return
    if _speak_with_espeak(text):
        return
    LOG.error("no TTS backend worked; install piper or espeak-ng")


def handle_assistant_audio(pcm: bytes) -> None:
    try:
        text = transcribe(pcm)
    except Exception as e:
        LOG.warning("assistant transcribe failed: %s", e)
        speak_text("Sorry, speech transcription failed.")
        return
    if not text:
        LOG.info("assistant: empty transcript")
        return
    LOG.info("assistant heard: %s", text)
    try:
        answer = ask_assistant(text)
    except Exception as e:
        LOG.warning("assistant call failed: %s", e)
        speak_text("Sorry, the local assistant failed.")
        return
    if not answer:
        LOG.warning("assistant returned no final answer")
        speak_text("Sorry, I did not get a final answer.")
        return
    LOG.info("assistant answer: %s", answer)
    speak_text(answer)


TYPE_BACKEND = os.environ.get("DICTATION_TYPER", "auto").lower()  # auto | wtype | ydotool | clipboard
ENABLE_BEEPS = os.environ.get("DICTATION_BEEPS", "1") not in ("0", "false", "no", "off")
BEEP_VOLUME = float(os.environ.get("DICTATION_BEEP_VOLUME", "0.15"))
DICTATION_SOCKET_PATH = Path(
    os.environ.get("DICTATION_SOCKET_PATH", str(Path.home() / ".cache" / "personal-assistant" / "dictation.sock"))
)
TYPE_LOCK = threading.Lock()


def send_chord_toggle() -> None:
    """Tell Talon to toggle its mic. Silent no-op if Talon isn't listening."""
    if not TALON_SOCK.exists():
        LOG.debug("chord: %s not present (Talon not running?); no-op", TALON_SOCK)
        return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(str(TALON_SOCK))
        s.sendall(b"toggle\n")
        s.close()
        LOG.info("chord: toggle sent to Talon")
    except OSError as e:
        LOG.warning("chord: send to %s failed: %s", TALON_SOCK, e)


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
    with TYPE_LOCK:
        _type_text_locked(text)


def _type_text_locked(text: str):
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


def _handle_ipc_client(conn: socket.socket) -> None:
    with conn:
        f = conn.makefile("rwb")
        with f:
            for raw_line in f:
                try:
                    msg = json.loads(raw_line.decode("utf-8"))
                    command = msg.get("command")
                    if command == "ping":
                        response = {"ok": True}
                    elif command == "type":
                        text = str(msg.get("text") or "")
                        LOG.info("IPC typing: %s", text)
                        type_text(text)
                        response = {"ok": True}
                    else:
                        response = {"ok": False, "error": f"unknown command {command!r}"}
                except Exception as e:
                    LOG.warning("IPC request failed: %s", e)
                    response = {"ok": False, "error": str(e)}
                f.write((json.dumps(response) + "\n").encode("utf-8"))
                f.flush()


def start_ipc_server() -> None:
    def serve() -> None:
        path = DICTATION_SOCKET_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(path))
        os.chmod(path, 0o600)
        srv.listen(8)
        LOG.info("dictation IPC listening: %s", path)
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_ipc_client, args=(conn,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()


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
    ap.add_argument(
        "--chord-key",
        default=os.environ.get("DICTATION_CHORD_KEY", "KEY_RIGHTSHIFT"),
        help="evdev key name that, held together with --key, fires the configured "
             "chord action (default $DICTATION_CHORD_KEY or KEY_RIGHTSHIFT). Pass "
             "empty string to disable the chord entirely.",
    )
    args = ap.parse_args()
    global KEY_PTT, KEY_CHORD
    try:
        KEY_PTT = getattr(evdev.ecodes, args.key)
    except AttributeError:
        LOG.error("unknown evdev key %r; check spelling (e.g. KEY_RIGHTCTRL, KEY_RIGHTSHIFT, KEY_SCROLLLOCK, KEY_F12)", args.key)
        return 2
    if args.chord_key:
        try:
            KEY_CHORD = getattr(evdev.ecodes, args.chord_key)
        except AttributeError:
            LOG.error("unknown evdev chord key %r", args.chord_key)
            return 2
    else:
        KEY_CHORD = None
    chord_action = os.environ.get("DICTATION_CHORD_ACTION", "assistant").strip().lower()
    if chord_action not in ("assistant", "talon", "none", "off", "disabled"):
        LOG.error("unknown DICTATION_CHORD_ACTION=%r (use: assistant|talon|none)", chord_action)
        return 2
    if chord_action in ("none", "off", "disabled"):
        KEY_CHORD = None
    if KEY_CHORD and chord_action == "assistant":
        chord_desc = f"; chord={args.chord_key}+{args.key}->assistant({ASSISTANT_URL})"
    elif KEY_CHORD:
        chord_desc = f"; chord={args.chord_key}+{args.key}->Talon({TALON_SOCK})"
    else:
        chord_desc = ""
    LOG.info("dictation daemon starting; PTT=%s; whisper=%s%s", args.key, WHISPER_URL, chord_desc)

    keyboards = find_keyboards()
    if not keyboards:
        return 1
    start_ipc_server()

    rec = Recorder()
    pressed = False           # PTT recording in progress
    press_t0 = 0.0
    assistant_pressed = False # assistant recording in progress
    assistant_t0 = 0.0
    ptt_down = False          # raw key state for PTT key
    chord_down = False        # raw key state for chord key (always False if KEY_CHORD is None)
    chord_fired = False       # latched True after the chord fires; resets when both keys release

    watched: Set[int] = {KEY_PTT} | ({KEY_CHORD} if KEY_CHORD else set())

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
                if ev.type != evdev.ecodes.EV_KEY or ev.code not in watched:
                    continue

                # Update raw key state (ignore key-repeat ev.value==2)
                if ev.value in (0, 1):
                    if ev.code == KEY_PTT:
                        ptt_down = (ev.value == 1)
                    elif KEY_CHORD is not None and ev.code == KEY_CHORD:
                        chord_down = (ev.value == 1)

                # Chord edge: both keys now held and we haven't fired this hold yet.
                # Fires once; latched until both keys are released.
                if KEY_CHORD is not None and ptt_down and chord_down and not chord_fired:
                    chord_fired = True
                    if chord_action == "assistant":
                        if pressed:
                            # Switch from direct dictation to assistant capture.
                            _ = rec.stop()
                            pressed = False
                            press_t0 = 0.0
                            LOG.info("chord: direct PTT cancelled by assistant chord")
                        assistant_pressed = True
                        assistant_t0 = time.monotonic()
                        LOG.info("assistant PTT down...")
                        play_tone(988, 70)
                        rec.start()
                    else:
                        if pressed:
                            # Cancel any in-flight PTT — drop the audio, no transcribe.
                            _ = rec.stop()
                            pressed = False
                            press_t0 = 0.0
                            play_tone(440, 50)
                            LOG.info("chord: PTT cancelled by chord")
                        LOG.info("chord: %s+%s -> Talon toggle", args.chord_key, args.key)
                        send_chord_toggle()
                    continue

                if assistant_pressed and (not ptt_down or not chord_down):
                    assistant_pressed = False
                    held = time.monotonic() - assistant_t0
                    play_tone(740, 90)
                    pcm = rec.stop()
                    if held < MIN_AUDIO_SEC:
                        LOG.info("assistant: too short, %.2fs <= %.2fs, ignored", held, MIN_AUDIO_SEC)
                        continue
                    LOG.info("assistant PTT up after %.2fs; processing %d bytes...", held, len(pcm))
                    handle_assistant_audio(pcm)
                    continue

                # Reset latch once both keys are physically released.
                if KEY_CHORD is not None and not ptt_down and not chord_down:
                    chord_fired = False

                # PTT logic — only on PTT key events, only when the chord isn't in play.
                if ev.code != KEY_PTT:
                    continue
                if ev.value == 1 and not pressed and not chord_down and not chord_fired:
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
