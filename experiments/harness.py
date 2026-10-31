"""Runs one controlled transfer: receiver process + proxy process + in-process sender.

Used by training/generate_dataset.py and experiments/run_experiments.py so
that training data and evaluation trials are produced in exactly the same way.

The receiver and proxy run as separate OS processes (not threads), so they do
not compete with the sender for Python's global interpreter lock and timing is
closer to separate machines.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from protocol.reliability import SenderConfig, TransferResult, send_file

ROOT = Path(__file__).resolve().parents[1]


def make_payload(path: str, size: int, seed: int) -> str:
    """Deterministic pseudo-random file (same seed -> same bytes)."""
    if not os.path.exists(path) or os.path.getsize(path) != size:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = random.Random(f"payload-{seed}")
        with open(path, "wb") as f:
            remaining = size
            while remaining:
                n = min(remaining, 1 << 20)
                f.write(rng.randbytes(n) if hasattr(rng, "randbytes")
                        else bytes(rng.getrandbits(8) for _ in range(n)))
                remaining -= n
    return path


def _start(args, cwd) -> Tuple[subprocess.Popen, int]:
    proc = subprocess.Popen([sys.executable] + args, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    line = proc.stdout.readline()          # programs print "READY <port>" when listening
    if not line.startswith("READY"):
        proc.kill()
        raise RuntimeError(f"failed to start {args[0]}: {line!r}")
    return proc, int(line.split()[1])


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _try_remove(path: str) -> None:
    """Clean up a temporary file; failing to delete it must not fail a transfer."""
    try:
        os.remove(path)
    except OSError:
        pass


class Lab:
    def __init__(self, workdir: str, receiver_port: int = 9500, proxy_port: int = 9501):
        self.workdir = os.path.abspath(workdir)
        self.out_dir = os.path.join(self.workdir, "received")
        self.receiver_port = receiver_port
        self.proxy_port = proxy_port
        self.receiver: Optional[subprocess.Popen] = None

    def __enter__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.receiver, _ = _start(["receiver.py", "--host", "127.0.0.1", "--port",
                                   str(self.receiver_port), "--output-dir", self.out_dir,
                                   "--log-level", "WARNING"], ROOT)
        return self

    def __exit__(self, *exc):
        if self.receiver:
            _stop(self.receiver)

    def run_trial(self, src: str, profile: str, seed: int, mode: str,
                  model_path: Optional[str] = None, obs_log: Optional[str] = None,
                  log_features: bool = False, progress=None, events=None,
                  **sender_overrides) -> Tuple[TransferResult, dict, bool]:
        """Returns (sender result, proxy ground-truth stats, independently verified file ok)."""
        stats_file = os.path.join(self.workdir, f"proxy_stats_{os.getpid()}.json")
        before = set(os.listdir(self.out_dir))
        proxy, _ = _start(["proxy.py", "--listen-port", str(self.proxy_port), "--target-port",
                           str(self.receiver_port), "--profile", profile, "--seed", str(seed),
                           "--stats-file", stats_file], ROOT)
        try:
            cfg = SenderConfig(port=self.proxy_port, timeout_mode=mode,
                               model_path=model_path or str(ROOT / "models" / "rto_model.json"),
                               obs_log=obs_log, log_features=log_features, **sender_overrides)
            result = send_file(src, cfg, progress=progress, events=events)
        finally:
            time.sleep(0.05)                 # let the proxy flush any in-flight packets
            _stop(proxy)
        with open(stats_file) as f:
            stats = json.load(f)
        _try_remove(stats_file)

        # Independent check: recompute SHA-256 of the file this transfer created
        # (compared with the folder listing taken before it started, so a file
        # left over from an earlier run can never be checked by mistake).
        new = [f for f in os.listdir(self.out_dir) if f not in before and not f.endswith(".part")]
        verified = False
        if len(new) == 1:
            received = os.path.join(self.out_dir, new[0])
            with open(src, "rb") as a, open(received, "rb") as b:
                verified = hashlib.sha256(a.read()).digest() == hashlib.sha256(b.read()).digest()
            _try_remove(received)
        return result, stats, verified
