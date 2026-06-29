# talon:latest
#
# Talon Voice in a Fedora 43 toolbox. Sits alongside whisper-rocm-7.2.3 — separate
# image so Talon iteration doesn't drag the 7 GB ROCm rebuild along.
#
# Tarball: supplied by the user. Download from https://talonvoice.com (login required),
# drop into ~/Downloads/, and run engine/build-talon.sh — it copies the latest
# talon-*-linux.tar.xz into the build context as `talon-linux.tar.xz` before invoking
# podman build, then removes it afterwards.
#
# Audio/display come for free: toolbox bind-mounts $XDG_RUNTIME_DIR (PipeWire socket)
# and $WAYLAND_DISPLAY into the container, so Talon's mic and status window work natively.
# Keystroke output goes through the host's ydotoold via /tmp/.ydotool_socket
# (also shared by toolbox).

FROM registry.fedoraproject.org/fedora:43

# Talon's runtime deps: X/Wayland client libs (Talon's UI is Qt-ish, links libxcb),
# audio client libs, GL, font/freetype for the status window, libffi for its bundled Python.
RUN --mount=type=cache,target=/var/cache/libdnf5,sharing=locked \
    --mount=type=cache,target=/var/cache/dnf,sharing=locked \
    dnf -y --nodocs --setopt=install_weak_deps=False --setopt=keepcache=True \
      install \
        bash ca-certificates sudo procps-ng \
        libxcb libxkbcommon libxkbcommon-x11 \
        libX11 libXi libXcursor libXdamage libXfixes libXrandr libXrender libXtst libXext \
        mesa-libGL mesa-libEGL \
        fontconfig freetype dejavu-sans-fonts \
        dbus-libs libffi \
        pipewire-libs pipewire-pulseaudio pulseaudio-libs alsa-lib portaudio \
        ydotool wl-clipboard wtype \
        curl jq vim-minimal git-core

# Tarball goes into the build context as talon-linux.tar.xz (build-talon.sh handles copy).
COPY talon-linux.tar.xz /tmp/talon.tar.xz
RUN mkdir -p /opt/talon \
 && tar -xJf /tmp/talon.tar.xz -C /opt/talon --strip-components=1 \
 && rm /tmp/talon.tar.xz

# Wrapper does what /opt/talon/run.sh does, minus the Tobii udev sudo step
# (no udev inside the toolbox). LD_LIBRARY_PATH picks up the bundled libpython3.11
# and Qt plugins from inside the tarball.
RUN cat > /usr/local/bin/talon <<'EOF' && chmod +x /usr/local/bin/talon
#!/bin/sh
exec env \
  LC_NUMERIC=C \
  QT_PLUGIN_PATH=/opt/talon/lib/plugins \
  LD_LIBRARY_PATH=/opt/talon/lib:/opt/talon/resources/python/lib:/opt/talon/resources/pypy/lib \
  /opt/talon/talon "$@"
EOF

LABEL org.opencontainers.image.title="talon" \
      org.opencontainers.image.description="Talon Voice in a Fedora 43 toolbox, sibling of whisper-strix-halo" \
      org.opencontainers.image.vendor="local" \
      org.opencontainers.image.licenses="proprietary (Talon Voice)"

CMD ["/bin/bash"]
