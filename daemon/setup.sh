#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"                       # daemon/
REPO="$(cd .. && pwd)"              # repo root (holds systemd/)

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*" >&2; }

step "Check for typing tools (wtype primary, ydotool fallback, wl-clipboard for clipboard mode)"
missing=()
command -v wtype     >/dev/null || missing+=(wtype)
command -v wl-copy   >/dev/null || missing+=(wl-clipboard)
command -v ydotool   >/dev/null || missing+=(ydotool)
command -v ydotoold  >/dev/null || true   # ydotoold ships in 'ydotool' on Fedora
if (( ${#missing[@]} )); then
  # wl-clipboard ships in the Fedora Silverblue base; never try to layer it.
  to_install=()
  for pkg in "${missing[@]}"; do
    [[ "$pkg" == "wl-clipboard" ]] && continue
    to_install+=("$pkg")
  done
  warn "missing host packages: ${missing[*]}"
  if (( ${#to_install[@]} )); then
    warn "Run this on the host, then reboot:"
    warn "   sudo rpm-ostree install ${to_install[*]}"
    warn "   systemctl reboot"
    warn "After reboot, re-run this script."
  fi
  if [[ " ${missing[*]} " == *" wl-clipboard "* ]]; then
    warn "(wl-clipboard should already be in the Silverblue base; check 'rpm -q wl-clipboard')"
  fi
  exit 1
fi
echo "wtype:   $(command -v wtype)   # primary, respects UK keyboard layout"
echo "ydotool: $(command -v ydotool) # fallback / used for Ctrl+Shift+V"
echo "wl-copy: $(command -v wl-copy)"

step "Check /dev/uinput permission (need write access)"
if [[ ! -w /dev/uinput ]]; then
  warn "/dev/uinput is not writable by you."
  warn "Installing udev rule to grant 'input' group access..."
  cat <<'EOF' | sudo tee /etc/udev/rules.d/60-uinput.rules >/dev/null
KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"
EOF
  sudo udevadm control --reload-rules
  sudo udevadm trigger /dev/uinput
  if ! id -nG | tr ' ' '\n' | grep -qx input; then
    sudo usermod -aG input "$USER"
    warn "added you to 'input' group. LOG OUT and back in (or reboot) for it to apply."
  fi
fi

step "Create venv with evdev + sounddevice + requests + numpy"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet evdev-binary sounddevice requests numpy

step "Install systemd-user units"
mkdir -p "$HOME/.config/systemd/user"
ln -sfv "$REPO/systemd/whisper-strix-halo.service" "$HOME/.config/systemd/user/whisper-strix-halo.service"
ln -sfv "$REPO/systemd/ydotoold.service"           "$HOME/.config/systemd/user/ydotoold.service"
ln -sfv "$REPO/systemd/dictation.service"          "$HOME/.config/systemd/user/dictation.service"
systemctl --user daemon-reload

step "Enable and start services"
systemctl --user enable --now whisper-strix-halo.service
systemctl --user enable --now ydotoold.service
systemctl --user enable --now dictation.service

step "Done. Status check:"
systemctl --user --no-pager status \
  whisper-strix-halo.service ydotoold.service dictation.service \
  | sed -n '1,5p;/Active:/p'

cat <<'EOF'

If you just added yourself to 'input' group, you need to log out + back in (or reboot)
before the daemon can read /dev/input/event*. After that, hold Right-Ctrl, speak,
release — your words should type into the focused window.

Logs:
   journalctl --user -u dictation.service -f
   journalctl --user -u whisper-strix-halo.service -f

EOF
