"""UDP impairment proxy: sits between sender and receiver and plays "the network".

    sender  --->  [listen socket]  proxy  [upstream socket]  --->  receiver
    sender  <---                                            <---  receiver

Every datagram is passed to Impairment.decide(), then dropped, delayed,
duplicated or corrupted. Delayed packets wait in a heap ordered by delivery
time; one loop with select() handles both sockets and the heap.

The proxy also records ground truth that the sender cannot observe, used
only for evaluation (never as ML features):
  * unnecessary retransmissions: a DATA retransmission although an earlier
    copy of the same chunk had already been delivered intact;
  * loss-recovery delay: time between a lost DATA copy and the next copy of the
    same chunk (= how long the sender's timer made it wait).
"""

from __future__ import annotations

import heapq
import json
import select
import socket
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from protocol.packet import MsgType, peek
from simulation.impairment import FWD, REV, Impairment
from simulation.profiles import Profile

BUFSIZE = 65535


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class Proxy:
    def __init__(self, listen: Tuple[str, int], target: Tuple[str, int], profile: Profile,
                 seed: int, status: bool = False):
        self.profile = profile
        self.seed = seed
        self.imp = Impairment(profile, seed)
        self.status = status
        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.client_sock.bind(listen)
        self.port = self.client_sock.getsockname()[1]
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_sock.connect(target)
        self.client_addr = None
        self.heap: List[tuple] = []           # (due, counter, direction, data)
        self._counter = 0
        self.t0: Optional[float] = None
        self.copies: Dict[tuple, int] = {}    # (direction, type, seq) -> copies seen
        # ground truth for DATA: seq -> list of (copy_no, arrival_time, delivered_intact)
        self.data_copies: Dict[int, List[tuple]] = {}
        self.counts = {d: {"received": 0, "dropped": 0, "delivered": 0, "duplicated": 0,
                           "corrupted": 0, "reordered": 0, "burst_drops": 0}
                       for d in ("fwd", "rev")}
        self._next_status = 0.0

    def elapsed(self, now: float) -> float:
        return 0.0 if self.t0 is None else now - self.t0

    # -- packet path -------------------------------------------------------
    def _on_packet(self, direction: int, data: bytes, now: float) -> None:
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        meta = peek(data)
        mtype, seq = meta if meta else (0, 0)
        key = (direction, mtype, seq)
        copy = self.copies.get(key, 0)
        self.copies[key] = copy + 1
        d = self.imp.decide(direction, mtype, seq, copy, t)
        c = self.counts["fwd" if direction == FWD else "rev"]
        c["received"] += 1
        if d.drop:
            c["dropped"] += 1
            if d.in_burst:
                c["burst_drops"] += 1
        else:
            if d.corrupt and len(data) > 0:
                pos = min(len(data) - 1, int(d.corrupt_pos * len(data)))
                data = data[:pos] + bytes([data[pos] ^ 0xFF]) + data[pos + 1:]
                c["corrupted"] += 1
            c["reordered"] += int(d.reordered)
            c["duplicated"] += int(d.duplicated)
            for delay in d.delays:
                self._counter += 1
                heapq.heappush(self.heap, (now + delay, self._counter, direction, data))
                c["delivered"] += 1
        if direction == FWD and mtype == MsgType.DATA:
            intact = (not d.drop) and (not d.corrupt)
            self.data_copies.setdefault(seq, []).append((copy, t, intact))

    def _deliver_due(self, now: float) -> None:
        while self.heap and self.heap[0][0] <= now:
            _, _, direction, data = heapq.heappop(self.heap)
            try:
                if direction == FWD:
                    self.server_sock.send(data)
                elif self.client_addr is not None:
                    self.client_sock.sendto(data, self.client_addr)
            except OSError:
                pass        # e.g. receiver not running: the packet is lost, like on a real network

    def _print_status(self, now: float) -> None:
        if not self.status or now < self._next_status:
            return
        self._next_status = now + 0.5
        t = self.elapsed(now)
        phase, delay, in_burst = self.imp.conditions(t)
        burst = " BURST" if in_burst else ""
        f, r = self.counts["fwd"], self.counts["rev"]
        sys.stderr.write(
            f"\r[proxy {self.profile.name}] t={t:6.1f}s one-way={delay:6.1f}ms "
            f"jitter={phase.jitter_ms:4.1f}ms loss={phase.loss_fwd * 100:4.1f}%{burst:6s} | "
            f"fwd {f['received']} rx {f['dropped']} drop | rev {r['received']} rx {r['dropped']} drop   ")
        sys.stderr.flush()

    # -- main loop ---------------------------------------------------------
    def run(self, stop: threading.Event) -> None:
        socks = [self.client_sock, self.server_sock]
        while not stop.is_set():
            now = time.monotonic()
            timeout = 0.2
            if self.heap:
                timeout = max(0.0, min(timeout, self.heap[0][0] - now))
            readable, _, _ = select.select(socks, [], [], timeout)
            for s in readable:
                try:
                    if s is self.client_sock:
                        data, addr = s.recvfrom(BUFSIZE)
                        self.client_addr = addr
                        self._on_packet(FWD, data, time.monotonic())
                    else:
                        data = s.recv(BUFSIZE)
                        self._on_packet(REV, data, time.monotonic())
                except (ConnectionRefusedError, ConnectionResetError):
                    pass
            now = time.monotonic()
            self._deliver_due(now)
            self._print_status(now)
        if self.status:
            sys.stderr.write("\n")
        self.client_sock.close()
        self.server_sock.close()

    # -- ground-truth statistics ---------------------------------------------
    def stats(self) -> dict:
        unnecessary = 0
        retx_copies = 0
        recovery: List[float] = []
        lost_copies = 0
        for seq, copies in self.data_copies.items():
            copies.sort()
            delivered_before = False
            for i, (copy, t, intact) in enumerate(copies):
                if copy > 0:
                    retx_copies += 1
                    if delivered_before:
                        unnecessary += 1
                if not intact:
                    lost_copies += 1
                    if i + 1 < len(copies):
                        recovery.append(copies[i + 1][1] - t)
                delivered_before = delivered_before or intact
        return {
            "profile": self.profile.name,
            "seed": self.seed,
            "counts": self.counts,
            "data_chunks_seen": len(self.data_copies),
            "data_retransmitted_copies": retx_copies,
            "unnecessary_retransmissions": unnecessary,
            "data_copies_lost": lost_copies,
            "loss_recovery_count": len(recovery),
            "loss_recovery_mean": (sum(recovery) / len(recovery)) if recovery else None,
            "loss_recovery_median": _pct(recovery, 50),
            "loss_recovery_p95": _pct(recovery, 95),
        }

    def write_stats(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.stats(), f, indent=2)
