#!/usr/bin/env bash
set -euo pipefail

PORT="${WHISPER_PORT:-8771}"
HOST="${WHISPER_HOST:-0.0.0.0}"
MODEL="${WHISPER_MODEL:-$HOME/Models/whisper-stt/ggml-large-v3-turbo.bin}"
THREADS="${WHISPER_THREADS:-8}"
# Prefer a host-side config the user can edit without rebuilding;
# fall back to the baked-in default if absent.
if [[ -z "${WHISPER_BRIDGE_CONFIG:-}" ]]; then
  if [[ -f "$HOME/.config/whisper-bridge/config.yaml" ]]; then
    CONFIG="$HOME/.config/whisper-bridge/config.yaml"
  else
    CONFIG="/etc/whisper-bridge/config.yaml"
  fi
else
  CONFIG="$WHISPER_BRIDGE_CONFIG"
fi
RUN_BRIDGE="${RUN_BRIDGE:-1}"

log() { printf '[whisper-toolbox] %s\n' "$*"; }

if [[ ! -f "$MODEL" ]]; then
  log "model not found at $MODEL"
  log "fetching with: whisper-download-model large-v3-turbo $(dirname "$MODEL")"
  mkdir -p "$(dirname "$MODEL")"
  whisper-download-model large-v3-turbo "$(dirname "$MODEL")"
fi

log "starting whisper-server on ${HOST}:${PORT} model=${MODEL}"
INFERENCE_PATH="${WHISPER_INFERENCE_PATH:-/v1/audio/transcriptions}"
log "inference endpoint: ${INFERENCE_PATH}  (OpenAI-compatible)"
whisper-server \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS" \
  --inference-path "$INFERENCE_PATH" &
WSERVER_PID=$!
BRIDGE_PID=""
cleanup_done=0
cleanup() {
  if [[ "$cleanup_done" -eq 1 ]]; then
    return
  fi
  cleanup_done=1
  log "shutting down"
  if [[ -n "${BRIDGE_PID:-}" ]]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
  fi
  kill "$WSERVER_PID" 2>/dev/null || true
  if [[ -n "${BRIDGE_PID:-}" ]]; then
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  wait "$WSERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log "waiting for whisper-server"
ready=0
for i in $(seq 1 60); do
  if ! kill -0 $WSERVER_PID 2>/dev/null; then
    log "ERROR: whisper-server exited before binding to :${PORT}. See log above."
    exit 1
  fi
  # demo page at / is independent of the inference path
  if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    log "whisper-server ready after ${i} polls; STT endpoint: http://${HOST}:${PORT}${INFERENCE_PATH}"
    ready=1
    break
  fi
  sleep 0.5
done
if [[ $ready -ne 1 ]]; then
  log "ERROR: whisper-server did not respond on :${PORT} after 30s. Aborting bridge."
  exit 1
fi

if [[ "$RUN_BRIDGE" != "1" ]]; then
  log "RUN_BRIDGE=0, leaving only whisper-server running"
  wait $WSERVER_PID
  exit $?
fi

log "starting bridge (config=${CONFIG})"
python3 /usr/local/lib/whisper-bridge/bridge.py --config "$CONFIG" &
BRIDGE_PID=$!
wait "$BRIDGE_PID"
