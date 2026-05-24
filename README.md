# personal-assistant

Hands-free voice tooling on top of a local STT/TTS stack for AMD Strix Halo
(gfx1151). Two halves:

- **`engine/`** — the container-side stack: a podman/toolbox image built on
  ROCm 7.2.3 + whisper.cpp, with `whisper-server` (OpenAI-compatible STT) and
  an optional wake-word bridge ("hey llama" / "hey hermes") that does TTS
  replies via Piper.
- **`daemon/`** — a host-side push-to-talk daemon (`dictation.py`): hold
  Right-Ctrl, speak, the transcript types into the focused window.

## How it fits

```
                ┌──────────────────────────────────────────────────┐
                │                  whisper-rocm-7.2.3 toolbox       │
                │  whisper-server :8771 (gfx1151 HIP, ~10x rt)      │
                │  piper-tts (en_GB-alba-medium)                    │
                └─────────────▲────────────────────────────────────┘
                              │  HTTP /v1/audio/transcriptions
                              │
   ┌──────────────────────────┴───────────────────────────┐
   │                                                       │
┌──┴────────────────────┐   ┌──────────────────────────────┴───┐
│ dictation.service     │   │ wake-word bridge (optional)       │
│ Right-Ctrl PTT →      │   │ "hey llama" / "hey hermes"        │
│   wtype into focused  │   │ → llama-server / Hermes VM        │
│   window              │   │ → piper voice reply               │
└───────────────────────┘   └──────────────────────────────────┘

Also reachable from: Open WebUI (mic button), Hermes Agent (Ctrl+B in VM).
```

## Files

| Path | What it is |
|---|---|
| `daemon/dictation.py` | The PTT daemon. evdev keyboard hook + sounddevice mic + POST to whisper-server + wtype/ydotool/clipboard. |
| `daemon/setup.sh` | Idempotent installer. Creates a venv, installs evdev/sounddevice/requests/numpy, lays down udev rule, registers + starts the three systemd-user services. |
| `daemon/.venv/` | Created by `setup.sh`. Python deps for `dictation.py`. |
| `engine/Dockerfile.whisper-rocm-7.2.3` | Toolbox image: Fedora 43 + ROCm 7.2.3 + gfx1151 HIP + whisper.cpp + Piper TTS. |
| `engine/build.sh` / `engine/create-toolbox.sh` | Build the image / register it as a toolbox with the right `/dev/dri`, `/dev/kfd`, `/dev/snd` device passthrough. |
| `engine/entrypoint.sh` | Launched inside the container; starts `whisper-server`, then (optionally) the bridge. |
| `engine/bridge.py` | Wake-word router: mic → whisper-server → wake-word match → `/v1/chat/completions` → Piper TTS reply. |
| `engine/config.default.yaml` | Default bridge config (wake phrases, agent endpoints, TTS voice). Copy to `~/.config/whisper-bridge/config.yaml` to override without rebuilding. |
| `systemd/whisper-strix-halo.service` | Autostarts the engine (whisper-server, plus bridge if `WHISPER_RUN_BRIDGE=1`) inside the `whisper-rocm-7.2.3` toolbox at login. |
| `systemd/ydotoold.service` | Runs `ydotoold` as your user so `ydotool` works without root. |
| `systemd/dictation.service` | Runs `daemon/dictation.py` from the venv. Depends on the above two. |

## Daemon: one-time install (host-level packages)

These need to live on the host because they touch `/dev/input/*` and `/dev/uinput`:

```sh
sudo rpm-ostree install wtype ydotool
systemctl reboot
```

Note: `wl-clipboard` is **already in the Fedora Silverblue base image**, so don't
include it in the install — rpm-ostree will refuse with "already provided by".
That's the desired state.

Why each:
- `wtype` — primary typing path. Layout-aware via Wayland virtual-keyboard protocol;
  respects your UK keymap (so `@`, `"`, `£`, `\`, `|`, `#` come out right).
- `ydotool` (and `ydotoold`) — fallback typing and Ctrl+Shift+V chords for clipboard mode.
- `wl-clipboard` — comes with the base image; provides `wl-copy` for the clipboard backend.

After reboot, run the setup script (idempotent — safe to re-run):

```sh
~/projects/personal-assistant/daemon/setup.sh
```

It will:
1. verify host packages are present;
2. install a `/etc/udev/rules.d/60-uinput.rules` so `/dev/uinput` is `input`-group writable;
3. add you to the `input` group if needed (then log out + back in);
4. create `.venv/` and pip-install `evdev sounddevice requests numpy`;
5. symlink the three systemd-user units into `~/.config/systemd/user/`;
6. `daemon-reload && enable --now` all three.

## Engine: build + run

The engine ships as a podman/toolbox image. Build once, then it's just another
systemd-user unit.

```sh
cd engine
./build.sh           # builds localhost/whisper-strix-halo:rocm-7.2.3 (~7 GB)
./create-toolbox.sh  # registers it as toolbox `whisper-rocm-7.2.3` with /dev/dri, /dev/kfd, /dev/snd
```

Models live on the host (the toolbox auto-mounts `$HOME`, so paths match inside
and outside the container):

- STT: `~/Models/whisper-stt/ggml-large-v3-turbo.bin` — auto-downloaded on
  first launch via `whisper-download-model`.
- TTS (only needed if you enable the bridge):
  `~/Models/tts/en/en_GB/alba/medium/en_GB-alba-medium.onnx` plus its `.onnx.json`.

The `whisper-strix-halo.service` unit auto-starts the container at login.
By default it runs **`whisper-server` only** (the OpenAI-compatible STT
endpoint on `:8771`). To also start the wake-word bridge, drop in:

```sh
systemctl --user edit whisper-strix-halo.service
# add:
[Service]
Environment=WHISPER_RUN_BRIDGE=1
```

Then `systemctl --user restart whisper-strix-halo.service`. The bridge reads
config from `~/.config/whisper-bridge/config.yaml` if present, else falls back
to `engine/config.default.yaml`. Edit `wake_words`, `agents.*.url`, `tts.voice_path`,
etc. to taste.

## Daily use

Hold **Right-Ctrl**, speak, release. The transcript types into whatever window has focus.

Audio cues:
- 880 Hz pip on press = "recording started"
- 660 Hz pip on release = "recording stopped, transcribing now"

Then the text appears (typically <1 s for short phrases, <3 s for long).

## Configuration

The daemon reads these environment variables (set them via `systemctl --user edit
dictation.service`):

| Variable | Default | What it does |
|---|---|---|
| `WHISPER_URL` | `http://127.0.0.1:8771/v1/audio/transcriptions` | Where to POST audio. Change if the whisper-server moves. |
| `WHISPER_LANG` | `en` | Whisper language hint. |
| `DICTATION_KEY` | `KEY_RIGHTCTRL` | evdev key name for PTT. Common picks: `KEY_RIGHTCTRL`, `KEY_RIGHTSHIFT`, `KEY_SCROLLLOCK`, `KEY_F12`, `KEY_CAPSLOCK`. Same effect as the `--key` CLI flag. |
| `DICTATION_TYPER` | `auto` | `auto` / `wtype` / `ydotool` / `clipboard`. Force a specific backend. |
| `TYPE_DELAY_MS` | `5` | Per-character delay for `ydotool type` (only used when ydotool is the backend). |
| `DICTATION_BEEPS` | `1` | `0` to silence the press/release tones. |
| `DICTATION_BEEP_VOLUME` | `0.15` | Tone volume (0.0 - 1.0). |
| `DICTATION_LOGLEVEL` | `INFO` | Python log level. `DEBUG` shows skipped input devices. |

CLI:
```
dictation.py --key KEY_RIGHTSHIFT     # use a different evdev key as PTT (overrides $DICTATION_KEY)
```

### Bridge (engine/)

The bridge reads `engine/config.default.yaml` (or `~/.config/whisper-bridge/config.yaml`
if it exists) and **expands `${VAR}` / `${VAR:-default}` references against the
process env before parsing the YAML**. So any field can be overridden from the
systemd unit without editing the config file:

```sh
systemctl --user edit whisper-strix-halo.service
# add:
[Service]
Environment=LLAMA_URL=http://other-host:5555/v1/chat/completions
Environment=HERMES_URL=http://hermes.vm:4444/v1/chat/completions
Environment=WHISPER_RUN_BRIDGE=1
```

Pre-wired env knobs (with sensible defaults baked in):

| Variable | Default | Field it overrides |
|---|---|---|
| `LLAMA_URL` | `http://127.0.0.1:3001/v1/chat/completions` | `agents.llama.url` |
| `HERMES_URL` | `http://REPLACE_ME:PORT/v1/chat/completions` | `agents.hermes.url` |
| `LLAMA_API_KEY` / `HERMES_API_KEY` | _(empty)_ | `agents.{name}.api_key`. If set, sent as `Authorization: Bearer <key>` — matches llama.cpp's `--api-key`, vLLM, and most hosted OpenAI-shape providers. |
| `LLAMA_MODEL` / `HERMES_MODEL` | `auto` | `agents.{name}.model` |
| `WHISPER_BRIDGE_URL` | `http://127.0.0.1:8771` | `whisper.url` |
| `WHISPER_LANG` | `en` | `whisper.language` |
| `BRIDGE_MODE` | `wake_word` | `mode` (`wake_word` / `vad_continuous` / `push_to_talk`) |
| `AUDIO_DEVICE` | `default` | `audio.device` |
| `TTS_ENABLED` | `true` | `tts.enabled` |
| `TTS_VOICE_PATH` | `~/Models/tts/en/en_GB/alba/medium/en_GB-alba-medium.onnx` | `tts.voice_path` |
| `DEFAULT_AGENT` | `hermes` | `default_agent` |

Anything else in the YAML can be templated the same way — wrap it in
`${YOUR_VAR_NAME:-fallback}` and set the env var to override.

**Wake phrases / activation commands** live under `wake_words:` in the YAML —
edit there to add aliases or new agents. The first phrase that matches a
transcript wins, so put longer / more specific aliases first. Each entry maps
to an `agents.<name>` block.

## Logs

```sh
journalctl --user -u dictation.service -f
journalctl --user -u whisper-strix-halo.service -f
journalctl --user -u ydotoold.service -f
```

## Common issues

**"no keyboards found"** — you're not in the `input` group. Run `id` to check;
re-run `setup.sh` (it'll add you) and log out + back in.

**Typing produces wrong characters** — `wtype` isn't being used (check the log for
which backend it picked). Likely fallback to `ydotool` which assumes US layout.
Confirm `wtype` is on `PATH`; if it is but failing, set
`DICTATION_TYPER=clipboard` to fall through to the layout-agnostic path.

**Nothing happens when I press Right-Ctrl** — check that `dictation.service` is
`active (running)`. Also check that something else (Mumble, Discord, a game)
isn't owning `Right-Ctrl` as a global hotkey.

**Audio is being recorded but transcript is empty** — usually means whisper-server
isn't reachable. Try `curl -sI http://127.0.0.1:8771/`. If that fails,
`systemctl --user restart whisper-strix-halo.service`.

**Typing into a terminal pastes literal `^V`** — the daemon used the clipboard
backend and the terminal interpreted Ctrl+Shift+V as something else. Set
`DICTATION_TYPER=wtype` to force direct typing.

## Stopping / disabling

```sh
systemctl --user stop dictation.service              # temporary
systemctl --user disable --now dictation.service    # autostart off
```

The whisper-server keeps running so Open WebUI and Hermes still have STT.

## Related

- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — upstream Dockerfiles `engine/Dockerfile.whisper-rocm-7.2.3` builds on
- `~/Models/whisper-stt/` — host-mounted STT models (downloaded on first run by `engine/entrypoint.sh`)
- `~/Models/tts/` — host-mounted Piper voices
- Open WebUI — can be pointed at the same `whisper-server` for its in-browser mic button via `AUDIO_STT_OPENAI_API_BASE_URL`
- Any OpenAI-compatible TUI/agent — set `STT_OPENAI_BASE_URL` to `http://<host>:8771/v1` to share the same STT brain
