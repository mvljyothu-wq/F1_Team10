"""Test helpers: run a receiver (and optionally the impairment proxy) in threads."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Optional

from protocol.reliability import Receiver, ReceiverConfig, SenderConfig, send_file
from simulation.profiles import get_profile
from simulation.proxy_core import Proxy


@contextmanager
def running_receiver(output_dir: str, **cfg):
    r = Receiver(ReceiverConfig(host="127.0.0.1", port=0, output_dir=output_dir, **cfg))
    stop = threading.Event()
    th = threading.Thread(target=r.serve, args=(stop,), daemon=True)
    th.start()
    try:
        yield r
    finally:
        stop.set()
        th.join(timeout=5)


@contextmanager
def running_proxy(target_port: int, profile: str, seed: int = 1):
    p = Proxy(("127.0.0.1", 0), ("127.0.0.1", target_port), get_profile(profile), seed)
    stop = threading.Event()
    th = threading.Thread(target=p.run, args=(stop,), daemon=True)
    th.start()
    try:
        yield p
    finally:
        stop.set()
        th.join(timeout=5)


def write_file(path: str, data: bytes) -> str:
    with open(path, "wb") as f:
        f.write(data)
    return path


def transfer(tmpdir: str, data: bytes, profile: Optional[str] = None, seed: int = 1,
             name: str = "payload.bin", receiver_kwargs=None, **sender_kwargs):
    """Send `data` through an optional proxy profile. Returns (result, receiver, proxy_stats)."""
    src = write_file(os.path.join(tmpdir, name), data)
    out = os.path.join(tmpdir, "out")
    with running_receiver(out, **(receiver_kwargs or {})) as r:
        if profile is None:
            res = send_file(src, SenderConfig(port=r.port, **sender_kwargs))
            return res, r, None
        with running_proxy(r.port, profile, seed) as p:
            res = send_file(src, SenderConfig(port=p.port, **sender_kwargs))
        return res, r, p.stats()
