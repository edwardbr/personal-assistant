#!/usr/bin/env python3
"""Audio bridge: mic -> whisper-server -> (wake-word router) -> OpenAI /v1/chat/completions."""
from __future__ import annotations

import argparse
import collections
import io
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import webrtcvad
import yaml

LOG = logging.getLogger("whisper-bridge")

FRAME_MS = 30
SAMPLE_WIDTH = 2  # int16


@dataclass
class AgentProfile:
    name: str
    url: str
    model: str
    system_prompt: str
    max_tokens: int
    temperature: float
    api_key: str = ""   # if set, sent as `Authorization: Bearer <key>` (llama.cpp --api-key)
    _resolved_model: Optional[str] = field(default=None, init=False, repr=False)

    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


@dataclass
class WakeWord:
    phrases: list[str]   # accept multiple spellings to ride out whisper mishearings
    agent: str

    @property
    def display(self) -> str:
        return self.phrases[0] if self.phrases else "(empty)"


@dataclass
class Cfg:
    mode: str
    whisper_url: str
    whisper_path: str
    whisper_language: str
    whisper_threads: int
    whisper_translate: bool
    tts_enabled: bool
    tts_voice_path: str
    tts_sample_rate: int
    audio_device: object
    sample_rate: int
    vad_aggressiveness: int
    silence_ms: int
    min_speech_ms: int
    wake_fuzzy: bool
    ack_beep: bool
    command_timeout_sec: float
    print_response: bool
    wake_words: list[WakeWord]
    agents: dict[str, AgentProfile]
    default_agent: str


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """Expand ${VAR} and ${VAR:-default} in a YAML source string before parsing.

    Lets users override any config field via env var (e.g. LLAMA_URL, HERMES_URL)
    without editing the YAML or rebuilding the image. Unmatched vars without a
    default expand to empty string, matching POSIX shell semantics.
    """
    def repl(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        return os.environ.get(var, default if default is not None else "")
    return _ENV_RE.sub(repl, text)


def load_config(path: str) -> Cfg:
    with open(path) as f:
        raw = yaml.safe_load(_expand_env(f.read()))

    agents: dict[str, AgentProfile] = {}
    for name, a in raw["agents"].items():
        agents[name] = AgentProfile(
            name=name,
            url=a["url"],
            model=a.get("model", "auto"),
            system_prompt=a.get("system_prompt", "You are a helpful assistant."),
            max_tokens=int(a.get("max_tokens", 256)),
            temperature=float(a.get("temperature", 0.7)),
            api_key=str(a.get("api_key", "") or ""),
        )

    wake_words: list[WakeWord] = []
    for w in raw.get("wake_words", []):
        if "phrases" in w:
            ph = [str(p).lower().strip() for p in w["phrases"] if str(p).strip()]
        else:
            ph = [str(w["phrase"]).lower().strip()]
        wake_words.append(WakeWord(phrases=ph, agent=w["agent"]))
    for w in wake_words:
        if w.agent not in agents:
            raise ValueError(f"wake_word {w.display!r} -> unknown agent {w.agent!r}")

    default_agent = raw.get("default_agent")
    if default_agent and default_agent not in agents:
        raise ValueError(f"default_agent {default_agent!r} not in agents")
    if not default_agent and agents:
        default_agent = next(iter(agents))

    wo = raw.get("wake_word_options", {})

    tts = raw.get("tts", {}) or {}
    tts_enabled = bool(tts.get("enabled", False))
    tts_voice_path = os.path.expanduser(os.path.expandvars(tts.get("voice_path", "")))
    tts_sample_rate = 22050
    if tts_enabled and tts_voice_path:
        json_path = tts_voice_path + ".json"
        try:
            with open(json_path) as f:
                tts_sample_rate = int(json.load(f).get("audio", {}).get("sample_rate", 22050))
        except Exception as e:
            LOG.warning("could not read sample rate from %s: %s (defaulting to 22050)", json_path, e)

    return Cfg(
        mode=raw.get("mode", "wake_word"),
        whisper_url=raw["whisper"]["url"].rstrip("/"),
        whisper_path=raw["whisper"].get("path", "/v1/audio/transcriptions"),
        whisper_language=raw["whisper"].get("language", "en"),
        whisper_threads=int(raw["whisper"].get("threads", 8)),
        whisper_translate=bool(raw["whisper"].get("translate", False)),
        audio_device=raw["audio"].get("device", "default"),
        sample_rate=int(raw["audio"].get("sample_rate", 16000)),
        vad_aggressiveness=int(raw["audio"].get("vad_aggressiveness", 2)),
        silence_ms=int(raw["audio"].get("silence_ms", 600)),
        min_speech_ms=int(raw["audio"].get("min_speech_ms", 300)),
        wake_fuzzy=bool(wo.get("fuzzy", True)),
        ack_beep=bool(wo.get("ack_beep", True)),
        command_timeout_sec=float(wo.get("command_timeout_sec", 12)),
        print_response=bool(wo.get("print_response", True)),
        wake_words=wake_words,
        agents=agents,
        default_agent=default_agent,
        tts_enabled=tts_enabled,
        tts_voice_path=tts_voice_path,
        tts_sample_rate=tts_sample_rate,
    )


def speak(cfg: Cfg, text: str):
    """Synthesize via piper CLI and play through PortAudio. Half-duplex (blocks loop)."""
    if not cfg.tts_enabled or not cfg.tts_voice_path:
        return
    text = text.strip()
    if not text:
        return
    try:
        proc = subprocess.run(
            ["piper", "-m", cfg.tts_voice_path, "--output-raw"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
            check=True,
        )
    except subprocess.TimeoutExpired:
        LOG.warning("TTS: piper timed out (>30s)")
        return
    except subprocess.CalledProcessError as e:
        LOG.warning("TTS: piper failed (exit %d): %s", e.returncode, (e.stderr or b"")[:200].decode("utf-8", "replace"))
        return
    except FileNotFoundError:
        LOG.warning("TTS: 'piper' not on PATH; disable tts in config or install piper-tts")
        return
    if not proc.stdout:
        LOG.warning("TTS: piper produced no audio")
        return
    audio = np.frombuffer(proc.stdout, dtype=np.int16)
    try:
        sd.play(audio, samplerate=cfg.tts_sample_rate, blocking=True)
    except Exception as e:
        LOG.warning("TTS playback failed: %s", e)


class SegmentCapture:
    """Captures speech segments from the mic using webrtcvad. Yields int16 PCM bytes per utterance."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.frame_samples = int(cfg.sample_rate * FRAME_MS / 1000)
        self.silence_frames = max(1, cfg.silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, cfg.min_speech_ms // FRAME_MS)
        self.stop_evt = threading.Event()

    def stop(self):
        self.stop_evt.set()

    def stream_segments(self):
        device = None if self.cfg.audio_device == "default" else self.cfg.audio_device
        try:
            with sd.RawInputStream(
                samplerate=self.cfg.sample_rate,
                blocksize=self.frame_samples,
                dtype="int16",
                channels=1,
                device=device,
            ) as stream:
                LOG.info("mic open: device=%s rate=%d", device or "default", self.cfg.sample_rate)
                yield from self._loop(stream)
        except sd.PortAudioError as e:
            LOG.error("PortAudio error: %s", e)
            LOG.error("Check that the toolbox can see PipeWire/Pulse: pactl info && arecord -l")
            raise

    def _loop(self, stream):
        ring: collections.deque = collections.deque(maxlen=int(0.3 * self.cfg.sample_rate / self.frame_samples))
        voiced: list[bytes] = []
        silent_run = 0
        triggered = False
        while not self.stop_evt.is_set():
            frame, overflowed = stream.read(self.frame_samples)
            if overflowed:
                LOG.warning("input overflow")
            if len(frame) < self.frame_samples * SAMPLE_WIDTH:
                continue
            try:
                is_speech = self.vad.is_speech(bytes(frame), self.cfg.sample_rate)
            except Exception:
                is_speech = False
            if not triggered:
                ring.append(bytes(frame))
                if is_speech:
                    triggered = True
                    voiced = list(ring)
                    voiced.append(bytes(frame))
                    silent_run = 0
            else:
                voiced.append(bytes(frame))
                if is_speech:
                    silent_run = 0
                else:
                    silent_run += 1
                    if silent_run >= self.silence_frames:
                        if len(voiced) >= self.min_speech_frames:
                            yield b"".join(voiced)
                        triggered = False
                        voiced = []
                        silent_run = 0
                        ring.clear()


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def transcribe(cfg: Cfg, pcm: bytes) -> str:
    wav = pcm_to_wav(pcm, cfg.sample_rate)
    files = {"file": ("clip.wav", wav, "audio/wav")}
    data = {
        "temperature": "0.0",
        "response_format": "json",
        "language": cfg.whisper_language,
        "threads": str(cfg.whisper_threads),
    }
    if cfg.whisper_translate:
        data["translate"] = "true"
    r = requests.post(f"{cfg.whisper_url}{cfg.whisper_path}", files=files, data=data, timeout=60)
    r.raise_for_status()
    j = r.json()
    return (j.get("text") or "").strip()


_WAKE_NORM = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return _WAKE_NORM.sub(" ", text.lower()).strip()


def find_wake_match(text: str, wake_words: list[WakeWord], fuzzy: bool):
    """Return (end_index_in_normalized_text, WakeWord) of the earliest matching wake phrase, or None.
    Tries every phrase variant in each WakeWord; first phrase to hit (earliest start) wins.
    """
    n = normalize(text)
    best: Optional[tuple[int, int, WakeWord]] = None  # (start, end, ww)
    for w in wake_words:
        for phrase in w.phrases:
            p = normalize(phrase)
            if not p:
                continue
            if not fuzzy:
                i = n.find(p)
                if i >= 0:
                    start, end = i, i + len(p)
                    if best is None or start < best[0]:
                        best = (start, end, w)
            else:
                pattern = r"\b" + r"[^a-z0-9]*".join(re.escape(part) for part in p.split()) + r"\b"
                m = re.search(pattern, n)
                if m:
                    if best is None or m.start() < best[0]:
                        best = (m.start(), m.end(), w)
    if best is None:
        return None
    return best[1], best[2]


def resolve_agent_model(agent: AgentProfile) -> str:
    """Pick a model for this agent.

    If `agent.model` is "auto", re-resolve on every call so we follow the server's
    currently-loaded model rather than caching a name that might be marked failed
    (common when llama.cpp's --models-preset swap fails on memory-tight models).
    """
    if agent.model != "auto":
        return agent.model
    base = agent.url.split("/v1/")[0]
    try:
        r = requests.get(f"{base}/v1/models", headers=agent.auth_headers(), timeout=5)
        r.raise_for_status()
        models = r.json().get("data", [])
        loaded = [m for m in models if (m.get("status", {}) or {}).get("value") == "loaded"]
        if loaded:
            chosen = loaded[0]["id"]
        else:
            healthy = [m for m in models if not (m.get("status", {}) or {}).get("failed")]
            chosen = (healthy or models)[0]["id"] if (healthy or models) else "default"
        if chosen != agent._resolved_model:
            LOG.info("[%s] model auto-resolved: %s", agent.name, chosen)
            agent._resolved_model = chosen
        return chosen
    except Exception as e:
        LOG.warning("[%s] could not resolve model: %s", agent.name, e)
    return agent._resolved_model or "default"


def call_agent(agent: AgentProfile, user_text: str) -> str:
    payload = {
        "model": resolve_agent_model(agent),
        "messages": [
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": agent.max_tokens,
        "temperature": agent.temperature,
        "stream": False,
    }
    r = requests.post(agent.url, json=payload, headers=agent.auth_headers(), timeout=120)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()


def beep():
    try:
        rate = 16000
        t = np.linspace(0, 0.12, int(rate * 0.12), endpoint=False)
        tone = (0.25 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        sd.play(tone, samplerate=rate, blocking=True)
    except Exception:
        pass


def handle_command(cfg: Cfg, agent: AgentProfile, user_text: str):
    LOG.info("[%s] >> %s", agent.name, user_text)
    try:
        reply = call_agent(agent, user_text)
    except Exception as e:
        LOG.error("[%s] agent call failed: %s", agent.name, e)
        return
    if cfg.print_response:
        print(f"\033[1;36m{agent.name}:\033[0m {reply}", flush=True)
    speak(cfg, reply)


def run_wake_word(cfg: Cfg, cap: SegmentCapture):
    phrases = "; ".join(
        f"{w.display}({len(w.phrases)} variants)->{w.agent}" for w in cfg.wake_words
    )
    LOG.info("mode=wake_word listening for %s", phrases)

    armed_until: float = 0.0
    armed_agent: Optional[AgentProfile] = None
    for pcm in cap.stream_segments():
        t0 = time.monotonic()
        try:
            text = transcribe(cfg, pcm)
        except Exception as e:
            LOG.warning("transcribe failed: %s", e)
            continue
        dt = time.monotonic() - t0
        if not text:
            continue
        LOG.info("heard (%.2fs): %s", dt, text)

        now = time.monotonic()
        armed = now < armed_until and armed_agent is not None

        if not armed:
            match = find_wake_match(text, cfg.wake_words, cfg.wake_fuzzy)
            if match is None:
                continue
            end, ww = match
            agent = cfg.agents[ww.agent]
            tail = normalize(text)[end:].strip()
            if tail:
                handle_command(cfg, agent, tail)
                continue
            armed_until = now + cfg.command_timeout_sec
            armed_agent = agent
            LOG.info("wake! (%s -> %s) awaiting command for %.1fs",
                     ww.display, agent.name, cfg.command_timeout_sec)
            if cfg.ack_beep:
                beep()
            continue

        # already armed -> this segment is the command for the armed agent
        agent = armed_agent
        armed_until = 0.0
        armed_agent = None
        handle_command(cfg, agent, text)


def run_vad_continuous(cfg: Cfg, cap: SegmentCapture):
    agent = cfg.agents[cfg.default_agent]
    LOG.info("mode=vad_continuous (every utterance -> %s)", agent.name)
    for pcm in cap.stream_segments():
        try:
            text = transcribe(cfg, pcm)
        except Exception as e:
            LOG.warning("transcribe failed: %s", e)
            continue
        if not text:
            continue
        LOG.info("heard: %s", text)
        handle_command(cfg, agent, text)


def run_push_to_talk(cfg: Cfg, cap: SegmentCapture):
    agent = cfg.agents[cfg.default_agent]
    LOG.info("mode=push_to_talk -> %s (send SIGUSR1 to capture next utterance, pid=%d)",
             agent.name, os.getpid())
    armed = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: armed.set())
    for pcm in cap.stream_segments():
        if not armed.is_set():
            continue
        armed.clear()
        try:
            text = transcribe(cfg, pcm)
        except Exception as e:
            LOG.warning("transcribe failed: %s", e)
            continue
        if not text:
            continue
        LOG.info("heard: %s", text)
        handle_command(cfg, agent, text)


def main():
    logging.basicConfig(
        level=os.environ.get("WHISPER_BRIDGE_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/etc/whisper-bridge/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    LOG.info("loaded config from %s (mode=%s, agents=%s, default=%s)",
             args.config, cfg.mode, list(cfg.agents), cfg.default_agent)

    cap = SegmentCapture(cfg)
    signal.signal(signal.SIGTERM, lambda *_: cap.stop())
    signal.signal(signal.SIGINT, lambda *_: cap.stop())

    if cfg.mode == "wake_word":
        run_wake_word(cfg, cap)
    elif cfg.mode == "vad_continuous":
        run_vad_continuous(cfg, cap)
    elif cfg.mode == "push_to_talk":
        run_push_to_talk(cfg, cap)
    else:
        LOG.error("unknown mode: %s", cfg.mode)
        sys.exit(2)


if __name__ == "__main__":
    main()
