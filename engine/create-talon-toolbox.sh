#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-talon}"
IMAGE="${IMAGE:-localhost/talon:latest}"

if toolbox list | awk '{print $2}' | grep -qx "$NAME"; then
  echo ">> removing existing toolbox $NAME"
  toolbox rm -f "$NAME"
fi

echo ">> creating toolbox $NAME from $IMAGE"
# Toolbox auto-shares: $HOME, $XDG_RUNTIME_DIR (PipeWire + Wayland sockets), /tmp, /dev/dri.
# Extra: /dev/snd for raw ALSA fallbacks, audio group for the PulseAudio cookie.
# We deliberately do NOT pass /dev/input — chord detection runs on the host
# (daemon/dictation.py), and Talon talks back to the host via the shared ydotoold socket.
toolbox create "$NAME" \
  --image "$IMAGE" \
  -- \
    --device /dev/snd \
    --group-add audio \
    --group-add sudo \
    --security-opt seccomp=unconfined

cat <<EOF
>> done. quick smoke test:
   toolbox enter $NAME
   # inside the toolbox:
   pactl info | head          # expect "Server Name: PulseAudio (on PipeWire ...)"
   echo \$WAYLAND_DISPLAY      # expect "wayland-0" (or similar)
   talon                      # should bring up the Talon status window on your desktop

   # if 'talon' starts cleanly, exit the toolbox and we'll wire up the
   # systemd unit + chord detection in Phase 2.
EOF
