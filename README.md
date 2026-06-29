# personal-assistant

Hands-free voice tooling on top of a local STT/TTS stack for AMD Strix Halo
(gfx1151). Two halves:

- **`engine/`** — the container-side stack: a podman/toolbox image built on
  ROCm 7.2.3 + whisper.cpp, with `whisper-server` (OpenAI-compatible STT) and
  a wake-word bridge ("hey llama" / "hey hermes") that does TTS replies via
  Piper.
- **`daemon/`** — a host-side push-to-talk daemon (`dictation.py`): hold
  Right-Ctrl, speak, the transcript types into the focused window. Hold
  Right-Ctrl + Right-Shift to ask the local llama.cpp assistant and hear a
  spoken answer, with no UI.

Optional add-on (see [Optional: Talon Voice](#optional-talon-voice) below):

- **Talon Voice** — sibling toolbox (`engine/Dockerfile.talon`) for voice
  *commands* (vs. prose dictation). Not required to use the rest of the
  project; the chord can still be pointed back at Talon with
  `DICTATION_CHORD_ACTION=talon`.

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
│ dictation.service     │   │ wake-word bridge                  │
│ Right-Ctrl PTT →      │   │ "hey llama" / "hey hermes"        │
│   wtype into focused  │   │ → llama-server / Hermes VM        │
│   window              │   │ → piper voice reply               │
│ Right-Ctrl+Shift →    │   │                                  │
│   llama.cpp → spoken  │   │                                  │
│   TTS answer          │   │                                  │
└───────────────────────┘   └──────────────────────────────────┘

Also reachable from: Open WebUI (mic button), Hermes Agent (Ctrl+B in VM).
```

## Files

| Path | What it is |
|---|---|
| `daemon/dictation.py` | The PTT daemon. evdev keyboard hook + sounddevice mic + POST to whisper-server + wtype/ydotool/clipboard. |
| `daemon/setup.sh` | Idempotent installer. Creates a venv, installs evdev/sounddevice/requests/numpy, lays down udev rule, registers + starts the three systemd-user services. |
| `daemon/.venv/` | Created by `setup.sh`. Python deps for `dictation.py`. |
| `engine/Dockerfile.whisper-rocm-7.2.3` | Toolbox image: Fedora 43 + ROCm 7.2.3 + gfx1151 HIP + whisper.cpp + Piper TTS. Includes `parec`/`paplay` for PipeWire/Pulse bridge capture and playback. |
| `engine/build.sh` / `engine/create-toolbox.sh` | Build the image / register it as a toolbox with the right `/dev/dri`, `/dev/kfd`, `/dev/snd` device passthrough. |
| `engine/entrypoint.sh` | Launched inside the container; starts `whisper-server`, then (optionally) the bridge. |
| `engine/bridge.py` | Wake-word router: mic → whisper-server → wake-word match → `/v1/chat/completions` → Piper TTS reply. |
| `engine/config.default.yaml` | Default bridge config (wake phrases, agent endpoints, TTS voice). Copy to `~/.config/whisper-bridge/config.yaml` to override without rebuilding. |
| `systemd/whisper-strix-halo.service` | Autostarts the engine (`whisper-server` plus the wake-word bridge) inside the `whisper-rocm-7.2.3` toolbox at login. |
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
In this personal setup it runs both **`whisper-server`** (the OpenAI-compatible
STT endpoint on `:8771`) and the wake-word bridge by default:

```ini
Environment=WHISPER_RUN_BRIDGE=1
```

Set `WHISPER_RUN_BRIDGE=0` in the unit for server-only mode. The bridge reads
config from `~/.config/whisper-bridge/config.yaml` if present, else falls back
to `engine/config.default.yaml`. Edit `wake_words`, `agents.*.url`,
`tts.voice_path`, etc. to taste.

On this machine the bridge uses the USB mic through PipeWire/Pulse:

```ini
Environment=AUDIO_DEVICE=pulse:alsa_input.usb-YC1006_YC1006-00.mono-fallback
Environment=AUDIO_SAMPLE_RATE=48000
```

The `pulse:<source-name>` form tells `engine/bridge.py` to record with `parec`
instead of raw PortAudio/ALSA, which avoids "device busy" failures when
PipeWire owns the microphone. Find or update the source name with:

```sh
pactl get-default-source
pactl list short sources
```

## Daily use

Hold **Right-Ctrl**, speak, release. The transcript types into whatever window has focus.

Hold **Right-Ctrl + Right-Shift**, speak, release either key. The daemon
transcribes what you said, sends it to the local llama.cpp endpoint at
`http://127.0.0.1:3001/v1/chat/completions`, and speaks the answer back through
Piper TTS. This path is headless; it does not use Talon or a browser UI.

Say **"hey lama"** or **"hey llama"** to use the wake-word bridge. If you say
only the wake phrase, wait for the beep and then speak the command within 12
seconds. You can also speak it as one sentence, for example:

```text
hey lama can you hear me
```

Say **"hey dictate"** to start continuous hands-free dictation into the
currently focused terminal, text box, or UI control. Each time you pause, the
bridge transcribes that segment and sends it to `dictation.service`, which
types it using the same backend as Right-Ctrl dictation. Say **"stop dictate"**
to leave this mode.

Audio cues:
- 880 Hz pip on press = "recording started"
- 660 Hz pip on release = "recording stopped, transcribing now"
- 988 Hz pip on assistant chord = "assistant recording started"
- 740 Hz pip on assistant release = "assistant is processing"
- wake bridge beep after "hey lama" / "hey llama" = "assistant is armed"
- wake bridge beep after "hey dictate" = "continuous dictation started"
- wake bridge beep after "stop dictate" = "continuous dictation stopped"

For Right-Ctrl, the text appears in the focused window (typically <1 s for
short phrases, <3 s for long). For the assistant chord, the answer is spoken
back instead.

## Configuration

The daemon reads these environment variables (set them via `systemctl --user edit
dictation.service`):

| Variable | Default | What it does |
|---|---|---|
| `WHISPER_URL` | `http://127.0.0.1:8771/v1/audio/transcriptions` | Where to POST audio. Change if the whisper-server moves. |
| `WHISPER_LANG` | `en` | Whisper language hint. |
| `DICTATION_KEY` | `KEY_RIGHTCTRL` | evdev key name for PTT. Common picks: `KEY_RIGHTCTRL`, `KEY_RIGHTSHIFT`, `KEY_SCROLLLOCK`, `KEY_F12`, `KEY_CAPSLOCK`. Same effect as the `--key` CLI flag. |
| `DICTATION_CHORD_KEY` | `KEY_RIGHTSHIFT` | Secondary key used with `DICTATION_KEY` for the assistant/Talon chord. Empty value disables the chord. |
| `DICTATION_CHORD_ACTION` | `assistant` | `assistant` sends speech to llama.cpp and speaks the answer; `talon` restores the old Talon toggle; `none` disables the chord. |
| `DICTATION_ASSISTANT_URL` | `http://127.0.0.1:3001/v1/chat/completions` | OpenAI-compatible chat endpoint for the headless assistant chord. |
| `DICTATION_ASSISTANT_MODEL` | `auto` | `auto` selects the currently loaded llama.cpp model from `/v1/models`; set a model id to force one. |
| `DICTATION_ASSISTANT_API_KEY` | _(empty)_ | Optional bearer token for the assistant endpoint. |
| `DICTATION_ASSISTANT_MAX_TOKENS` | `256` | Maximum assistant response tokens. |
| `DICTATION_ASSISTANT_TEMPERATURE` | `0.2` | Assistant response temperature. |
| `DICTATION_ASSISTANT_DISABLE_THINKING` | `1` | Sends `chat_template_kwargs.enable_thinking=false` for Qwen-style llama.cpp models so TTS reads only the final answer. |
| `DICTATION_ASSISTANT_TTS` | `1` | `0` disables spoken assistant responses. |
| `DICTATION_TTS_VOICE_PATH` | `~/Models/tts/en/en_GB/alba/medium/en_GB-alba-medium.onnx` | Piper voice used for spoken assistant responses. |
| `DICTATION_TTS_TOOLBOX` | `whisper-rocm-7.2.3` | Toolbox used for Piper if `piper` is not installed on the host. |
| `DICTATION_TTS_ESPEAK_VOICE` | `en-gb` | Fallback `espeak-ng` voice if Piper is unavailable. |
| `DICTATION_TYPER` | `direct` | `auto` / `direct` / `wtype` / `ydotool` / `clipboard`. `direct` avoids the clipboard and uses direct key injection (`wtype` where supported, otherwise `ydotool`). |
| `DICTATION_ALLOW_CLIPBOARD_FALLBACK` | `0` | When `0`, direct typing failures do not copy text to the clipboard. Set to `1` if you want clipboard fallback. |
| `DICTATION_PASTE_KEYS` | `ctrl_v` | Clipboard backend paste shortcut. Use `ctrl_v` for editors and most GUI text fields; use `ctrl_shift_v` for terminals; `shift_insert` is also supported. |
| `DICTATION_SOCKET_PATH` | `~/.cache/personal-assistant/dictation.sock` | Unix socket used by the wake bridge's `hey dictate` mode to ask `dictation.service` to type text into the focused window. |
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
| `AUDIO_DEVICE` | `default` | `audio.device`. Use `pulse:<source-name>` for PipeWire/Pulse capture through `parec`; this machine uses `pulse:alsa_input.usb-YC1006_YC1006-00.mono-fallback`. |
| `AUDIO_SAMPLE_RATE` | `48000` | `audio.sample_rate`. `48000` matches PipeWire and avoids the invalid-sample-rate errors seen with this USB mic. |
| `AUDIO_MIN_RMS_DBFS` | `-42` | `audio.min_rms_dbfs`. Segments quieter than this are dropped before Whisper to reduce background-noise hallucinations. Raise it toward `-38` if noise still leaks through; lower it toward `-48` if quiet speech is missed. |
| `DICTATION_SOCKET_PATH` | `~/.cache/personal-assistant/dictation.sock` | `dictation.socket_path`. Must match the daemon setting for `hey dictate`. |
| `MCP_ENABLED` | `true` | `mcp.enabled`. Enables MCP tools for assistant wake routes such as `hey lama`. |
| `TTS_ENABLED` | `true` | `tts.enabled` |
| `TTS_VOICE_PATH` | `~/Models/tts/en/en_GB/alba/medium/en_GB-alba-medium.onnx` | `tts.voice_path` |
| `DEFAULT_AGENT` | `hermes` | `default_agent` |

Anything else in the YAML can be templated the same way — wrap it in
`${YOUR_VAR_NAME:-fallback}` and set the env var to override.

**Wake phrases / activation commands** live under `wake_words:` in the YAML —
edit there to add aliases or new agents. The first phrase that matches a
transcript wins, so put longer / more specific aliases first. Each entry maps
to an `agents.<name>` block, except `action: dictate`, which starts continuous
typing through `dictation.service` until a configured stop phrase is heard.

For Qwen-style llama.cpp models, the bridge sends
`chat_template_kwargs.enable_thinking=false` by default so Piper reads only the
final answer, not an empty response with hidden reasoning.

The default agent prompts are written for spoken output: they ask the model to
avoid Markdown, bullet markers, asterisks, tables, and code fences. The TTS path
also strips common Markdown markers before sending text to Piper, so a bullet
list is read as plain spoken sentences rather than as "asterisk" punctuation.

The bridge also filters exact normalized transcripts listed in
`wake_word_options.ignored_transcripts` before wake matching or dictation. This
is where common Whisper-on-noise hallucinations such as `thank you` are blocked.
Repeated ignored phrases, such as `thank you thank you`, and repeated `beep`
transcripts are filtered as well. If a dictation-start cue leaks into the same
transcript as your first dictated words, the leading `beep` is stripped before
typing.

### MCP Tools

The assistant wake routes (`hey lama`, `hey hermes`) can expose stdio MCP tools
to the local model. The bundled local MCP server is enabled by default and
provides:

- `get_time` — current local date/time, with optional IANA timezone.
- `web_search` — concise internet lookup via public web/Wikipedia endpoints.
- `news_headlines` — current headlines from RSS feeds, with UK/world/topic
  selection.

Examples:

```text
hey lama what time is it
hey lama look up the latest AMD Strix Halo news
hey lama what are the news headlines in the United Kingdom
```

Add more stdio MCP servers under `mcp.servers` in
`~/.config/whisper-bridge/config.yaml`:

```yaml
mcp:
  enabled: true
  max_tool_rounds: 3
  servers:
    local:
      command: python3
      args:
        - /usr/local/lib/whisper-bridge/local_mcp_server.py
    my_server:
      command: /path/to/mcp-server
      args: ["--flag", "value"]
```

## Optional: Talon Voice

Talon (talonvoice.com) is a voice-command engine — different shape from the
Whisper dictation here: you speak structured commands ("camel my var name",
"go line forty two") and Talon's grammar layer composes the keystrokes.

This is **optional and additive** — none of the core dictation stack depends
on it. Skip this section unless you specifically want command-mode voice
control.

### Phase 1 — container

1. Sign up at <https://talonvoice.com> (free), grab the latest
   Linux x86_64 tarball, save it as `~/Downloads/talon*linux*.tar.xz`
   (any version-suffix or none works; the newest matching file wins).
2. Build and register the toolbox:
   ```sh
   cd engine
   ./build-talon.sh           # builds localhost/talon:latest (~250 MB)
   ./create-talon-toolbox.sh  # registers toolbox `talon` with /dev/snd
   ```
3. Smoke-test:
   ```sh
   toolbox enter talon
   talon                      # status window should appear on your desktop
   ```

Talon is closed-source but free to use. The community config
(`talonhub/community`) is MIT and lives in `~/.talon/user/community/` once
you clone it in there.

### Phase 2 — chord activation

`daemon/dictation.py` uses **Right-Ctrl + Right-Shift** for the headless
llama.cpp assistant by default. To use that chord for Talon instead, set:

```ini
[Service]
Environment=DICTATION_CHORD_ACTION=talon
```

With that override, holding both keys sends a `toggle` command over
`~/.talon/chord.sock` to a small user script
(`engine/talon-user/chord_listener.py`) loaded by Talon, which calls
`actions.speech.toggle()`. Right-Ctrl alone still triggers prose dictation
exactly as before; if Talon isn't running, the Talon chord is a silent no-op.

Wire it up (after Phase 1 has produced `~/.talon/`):

```sh
cd engine
./install-talon-integration.sh
```

That script:
1. Symlinks `engine/talon-user/chord_listener.py` into `~/.talon/user/`
2. Symlinks `systemd/talon.service` into `~/.config/systemd/user/`
3. `enable --now` for `talon.service`
4. Restarts `dictation.service` so it picks up the chord defaults

Disable the chord without uninstalling Talon: `systemctl --user edit
dictation.service` and add `Environment=DICTATION_CHORD_KEY=` (empty value).

Files added by this phase:

| Path | What it is |
|---|---|
| `engine/talon-user/chord_listener.py` | Talon user script: Unix-socket listener that calls `actions.speech.{toggle,enable,disable}` on `toggle` / `wake` / `sleep`. |
| `systemd/talon.service` | Autostarts Talon inside the `talon` toolbox at login, mirroring `whisper-strix-halo.service`. |
| `engine/install-talon-integration.sh` | One-shot wiring: symlinks + `systemctl enable --now`. |

## Logs

```sh
journalctl --user -u dictation.service -f
journalctl --user -u whisper-strix-halo.service -f
journalctl --user -u ydotoold.service -f
```

## Common issues

**"no keyboards found"** — you're not in the `input` group. Run `id` to check;
re-run `setup.sh` (it'll add you) and log out + back in.

**Typing produces wrong characters** — direct `ydotool type` assumes a US-style
layout. The clipboard backend is layout-agnostic, but writes transcripts to the
clipboard. Set `DICTATION_TYPER=clipboard` if layout correctness matters more
than avoiding the clipboard.

**Nothing happens when I press Right-Ctrl** — check that `dictation.service` is
`active (running)`. Also check that something else (Mumble, Discord, a game)
isn't owning `Right-Ctrl` as a global hotkey.

**Audio is being recorded but transcript is empty** — usually means whisper-server
isn't reachable. Try `curl -sI http://127.0.0.1:8771/`. If that fails,
`systemctl --user restart whisper-strix-halo.service`.

**"Hey lama" is recognized but you hear nothing** — check
`journalctl --user -u whisper-strix-halo.service -f`. If the log shows
`wake!` / `[llama] >>` but no sound, playback is the issue. The bridge uses
`paplay` for the wake beep and Piper TTS, so verify `pactl info` works inside
the toolbox and that `AUDIO_SAMPLE_RATE=48000`.

**Wake bridge mic opens with `PortAudio error` or `Device or resource busy`** —
set `AUDIO_DEVICE=pulse:<source-name>` in `systemd/whisper-strix-halo.service`
and restart it. Get the source name with `pactl get-default-source`. This repo's
unit is currently set to the USB mic source on this machine.

**Wake bridge keeps hearing `Thank you` or other background-noise text** — the
bridge filters common hallucinations in `wake_word_options.ignored_transcripts`
and uses balanced defaults (`vad_aggressiveness: 2`, `min_speech_ms: 350`,
`min_rms_dbfs: -42`). If noise still leaks through, raise `min_rms_dbfs` toward
`-38` or add the exact unwanted phrase to `ignored_transcripts`.

**Dictation works in a terminal but not a text editor** — this repo's service
defaults to `DICTATION_TYPER=direct`, which avoids the clipboard and should type
into both terminals and editors through `ydotool`. If you switch back to the
clipboard backend on GNOME Wayland, text editors expect `Ctrl+V`, while terminals
expect `Ctrl+Shift+V`; control that with `DICTATION_PASTE_KEYS`.

**Typing into a terminal pastes literal `^V` or does not paste** — set
`DICTATION_PASTE_KEYS=ctrl_shift_v`, or set `DICTATION_TYPER=wtype` on a
compositor that supports `wtype`.

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
