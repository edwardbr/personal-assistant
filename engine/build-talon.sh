#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-localhost/talon:latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.talon}"
TARBALL_GLOB="${TARBALL_GLOB:-$HOME/Downloads/talon*linux*.tar.xz}"

cd "$(dirname "$0")"

# Pick the newest matching tarball.
shopt -s nullglob
matches=( $TARBALL_GLOB )
shopt -u nullglob
if [[ ${#matches[@]} -eq 0 ]]; then
  cat >&2 <<EOF
ERROR: no Talon tarball found matching: $TARBALL_GLOB
  Download from https://talonvoice.com (free account required, then Downloads page),
  pick "Linux x86_64", and save the .tar.xz into ~/Downloads/ (any filename
  matching talon*linux*.tar.xz works; the newest by sort order wins).
EOF
  exit 1
fi
TARBALL=$(printf '%s\n' "${matches[@]}" | sort -V | tail -1)
echo ">> using tarball: $TARBALL"

# Copy (don't move) so re-runs are safe. Cleaned up on exit even if build fails.
cp "$TARBALL" ./talon-linux.tar.xz
trap 'rm -f ./talon-linux.tar.xz' EXIT

echo ">> building $IMAGE from $DOCKERFILE"
podman build \
  --layers \
  --tag "$IMAGE" \
  --file "$DOCKERFILE" \
  .

echo ">> done: $IMAGE"
podman images "$IMAGE"
