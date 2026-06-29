"""Chord listener: external IPC for toggling Talon's mic.

Listens on a Unix socket at ~/.talon/chord.sock. Each accepted connection
reads one short line. Recognised commands:

    toggle   actions.speech.toggle()
    wake     actions.speech.enable()
    sleep    actions.speech.disable()

The host-side daemon/dictation.py writes "toggle\\n" to this socket when it
sees a Right-Ctrl + Right-Shift chord. The socket lives inside $HOME so it
is visible to both the host (where dictation.py runs) and the toolbox
container (where Talon runs) — toolbox bind-mounts $HOME by default.

This file is installed by symlinking it into ~/.talon/user/ (see
engine/install-talon-integration.sh).
"""
from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path

from talon import actions, app

LOG = logging.getLogger("user.chord_listener")

SOCKET_PATH = Path.home() / ".talon" / "chord.sock"


def _handle(line: str) -> None:
    cmd = line.strip().lower()
    if cmd == "toggle":
        actions.speech.toggle()
    elif cmd == "wake":
        actions.speech.enable()
    elif cmd == "sleep":
        actions.speech.disable()
    else:
        LOG.warning("chord_listener: unknown command %r", cmd)


def _serve() -> None:
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        try:
            SOCKET_PATH.unlink()
        except OSError as e:
            LOG.error("chord_listener: cannot remove stale socket %s: %s", SOCKET_PATH, e)
            return
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(SOCKET_PATH))
        os.chmod(str(SOCKET_PATH), 0o600)
        sock.listen(4)
        LOG.info("chord_listener: listening on %s", SOCKET_PATH)
        while True:
            try:
                conn, _ = sock.accept()
            except OSError as e:
                LOG.warning("chord_listener: accept failed: %s", e)
                continue
            with conn:
                try:
                    data = conn.recv(64)
                except OSError:
                    continue
                if data:
                    _handle(data.decode("ascii", errors="replace"))
    finally:
        sock.close()
        if SOCKET_PATH.exists():
            try:
                SOCKET_PATH.unlink()
            except OSError:
                pass


def _on_ready() -> None:
    threading.Thread(target=_serve, daemon=True, name="chord-listener").start()


app.register("ready", _on_ready)
