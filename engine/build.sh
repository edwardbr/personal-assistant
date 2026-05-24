#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-localhost/whisper-strix-halo:rocm-7.2.3}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.whisper-rocm-7.2.3}"

cd "$(dirname "$0")"

echo ">> building $IMAGE from $DOCKERFILE"
podman build \
  --layers \
  --tag "$IMAGE" \
  --file "$DOCKERFILE" \
  .
echo ">> done: $IMAGE"
podman images "$IMAGE"
