#!/usr/bin/env bash
# Wire the optional Talon integration into the host:
#   - symlink chord_listener.py into ~/.talon/user/  (so Talon loads it on next start)
#   - symlink systemd/talon.service into ~/.config/systemd/user/
#   - enable + start talon.service
#
# Talon must already be installed and run at least once (so ~/.talon exists).
# If you haven't done that yet, run engine/build-talon.sh and engine/create-talon-toolbox.sh
# first, then `toolbox enter talon && talon` to accept the EULA.
#
# Restarting dictation.service afterwards picks up the chord defaults (KEY_RIGHTSHIFT
# alongside KEY_RIGHTCTRL); without Talon listening the chord is a silent no-op.

set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"                # engine/
REPO="$(cd .. && pwd)"       # repo root (holds systemd/)

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }

if [[ ! -d "$HOME/.talon" ]]; then
  warn "~/.talon not found. Run 'toolbox enter talon && talon' once to bootstrap it, then re-run."
  exit 1
fi

step "Symlink chord_listener.py into ~/.talon/user/"
mkdir -p "$HOME/.talon/user"
ln -sfv "$HERE/talon-user/chord_listener.py" "$HOME/.talon/user/chord_listener.py"

step "Symlink talon.service into ~/.config/systemd/user/"
mkdir -p "$HOME/.config/systemd/user"
ln -sfv "$REPO/systemd/talon.service" "$HOME/.config/systemd/user/talon.service"
systemctl --user daemon-reload

step "Enable and start talon.service"
systemctl --user enable --now talon.service

step "Restart dictation.service so it picks up chord defaults"
if systemctl --user is-enabled --quiet dictation.service; then
  systemctl --user restart dictation.service
else
  warn "dictation.service is not enabled; chord won't fire until you run daemon/setup.sh"
fi

step "Status:"
systemctl --user --no-pager status talon.service dictation.service \
  | sed -n '1,5p;/Active:/p'

cat <<'EOF'

Talon should be running inside the `talon` toolbox; the chord listener
will start with it. Tap Right-Ctrl + Right-Shift to toggle Talon's mic;
hold Right-Ctrl alone for prose dictation as before.

Logs:
   journalctl --user -u talon.service -f
   journalctl --user -u dictation.service -f
   tail -F ~/.talon/talon.log
EOF
