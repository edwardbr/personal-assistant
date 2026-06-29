#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-whisper-rocm-7.2.3}"
IMAGE="${IMAGE:-localhost/whisper-strix-halo:rocm-7.2.3}"

if toolbox list | awk '{print $2}' | grep -qx "$NAME"; then
  echo ">> removing existing toolbox $NAME"
  toolbox rm -f "$NAME"
fi

echo ">> creating toolbox $NAME from $IMAGE"
# Flags mirror kyuz0's rocm-7.2.3 launch; toolbox already bind-mounts $XDG_RUNTIME_DIR,
# so the PipeWire socket reaches inside without extra plumbing.
toolbox create "$NAME" \
  --image "$IMAGE" \
  -- \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/snd \
    --group-add video \
    --group-add render \
    --group-add audio \
    --group-add sudo \
    --security-opt seccomp=unconfined

cat <<EOF
>> done. next steps:
   toolbox enter $NAME
   # inside the toolbox:
   rocminfo | grep -i gfx        # expect gfx1151
   pactl info | head             # expect "Server Name: PulseAudio (on PipeWire ...)"
   pactl get-default-source      # expect your microphone source
   pactl list short sources      # use pulse:<source-name> for AUDIO_DEVICE
   whisper-toolbox-start         # starts whisper-server on :8771 and the wake-word bridge

   # from your host browser (or any other toolbox), the test web UI is at:
   #   http://localhost:8771/
EOF
