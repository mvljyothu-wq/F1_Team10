"""Selective-Repeat window bookkeeping, with no sockets and no clock.

SendWindow  -- what the sender has in flight, per-packet timers, ACK handling,
               and optional time-based fast loss detection (RACK, RFC 8985).
ReceiveBuffer -- which chunks the receiver has, out-of-order buffering.

Both take explicit `now` arguments so they can be unit-tested deterministically.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class InFlight:
    """One unacknowledged DATA packet."""
    seq: int
    first_sent: float          # time of the original transmission
    last_sent: float           # time of the latest (re)transmission
    deadline: float            # when its retransmission timer expires
    retx: int = 0              # number of retransmissions so far
    epoch: int = 0             # timeout manager's backoff epoch when last sent
    fast_done: bool = False    # declared lost by RACK for its current transmission
    rack_resent: bool = False  # latest transmission was a RACK retransmission


class SendWindow:
    """Sender side of Selective Repeat.

    Invariants:
      base      = lowest chunk not yet acknowledged
      next_seq  = next chunk that has never been sent
      base <= next_seq <= base + window

    Fast loss detection (rack=True) follows RACK (RFC 8985): remember the send
    time of the most recently *sent* packet that has been delivered, and that
    packet's RTT. Any packet still in flight that was sent before it and has
    been outstanding longer than RTT + reordering window is presumed lost and
    reported by detect_losses(), without waiting for its retransmission timer.
    The reordering window (min RTT / 4, as in the RFC) stops ordinary jitter
    from looking like loss. It adapts: when an ACK for a RACK-resent packet
    arrives sooner than any real round trip could (so it acknowledges the
    original copy, which was only reordered), the window grows by another
    min RTT / 4, up to one RTT. After 16 loss repairs without such a mistake
    it returns to the default (RFC 8985 section 6.2, using this signal in
    place of TCP's DSACK). A simpler "3 later ACKs" rule was tried first and
    rejected: under per-packet jitter it fired thousands of spurious resends.
    """

    def __init__(self, total_chunks: int, window: int, rack: bool = False,
                 reo_wnd_frac: float = 0.25):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.total = total_chunks
        self.window = window
        self.rack = rack
        self.reo_wnd_frac = reo_wnd_frac
        self.rack_xmit: Optional[float] = None    # send time of the newest-sent delivered packet
        self.rack_rtt: Optional[float] = None     # RTT of that packet
        self.min_rtt: Optional[float] = None
        self.reo_mult = 1                         # reordering window = reo_mult x frac x min RTT
        self.spurious_detected = 0
        self._repairs_since_spurious = 0
        self.base = 0
        self.next_seq = 0
        self.inflight: Dict[int, InFlight] = {}
        self.acked = bytearray(total_chunks)    # 1 byte per chunk
        self._timers: List[Tuple[float, int]] = []   # heap of (deadline, seq)

    # -- sending ---------------------------------------------------------
    def can_send_new(self) -> bool:
        return self.next_seq < self.total and self.next_seq < self.base + self.window

    def on_new_send(self, now: float, deadline: float, epoch: int) -> int:
        """Register the first transmission of chunk next_seq; returns its seq."""
        if not self.can_send_new():
            raise RuntimeError("window full or all chunks sent")
        seq = self.next_seq
        self.next_seq += 1
        self.inflight[seq] = InFlight(seq, now, now, deadline, 0, epoch)
        heapq.heappush(self._timers, (deadline, seq))
        return seq

    def on_retransmit(self, seq: int, now: float, deadline: float, epoch: int,
                      by_rack: bool = False) -> InFlight:
        entry = self.inflight[seq]
        entry.retx += 1
        entry.last_sent = now
        entry.deadline = deadline
        entry.epoch = epoch
        entry.fast_done = False
        entry.rack_resent = by_rack
        heapq.heappush(self._timers, (deadline, seq))
        return entry

    def rearm_all(self, rto: float, epoch: int) -> None:
        """Restart every pending timer as last_sent + rto (used when the
        timeout shrinks after a backoff ends, so no packet keeps a stale,
        backed-off timer)."""
        self._timers = []
        for entry in self.inflight.values():
            entry.deadline = entry.last_sent + rto
            entry.epoch = epoch
            self._timers.append((entry.deadline, entry.seq))
        heapq.heapify(self._timers)

    # -- timers ----------------------------------------------------------
    def _discard_stale_timers(self) -> None:
        # Heap entries become stale when a packet is ACKed or re-armed;
        # they are skipped lazily instead of being removed from the heap.
        while self._timers:
            deadline, seq = self._timers[0]
            entry = self.inflight.get(seq)
            if entry is not None and entry.deadline == deadline:
                return
            heapq.heappop(self._timers)

    def next_deadline(self) -> Optional[float]:
        self._discard_stale_timers()
        return self._timers[0][0] if self._timers else None

    def pop_expired(self, now: float) -> List[InFlight]:
        """Return in-flight packets whose timer has expired, earliest first."""
        expired = []
        while True:
            self._discard_stale_timers()
            if not self._timers or self._timers[0][0] > now:
                return expired
            _, seq = heapq.heappop(self._timers)
            expired.append(self.inflight[seq])

    # -- acknowledgements --------------------------------------------------
    def on_ack(self, seq: int, cum_ack: int, now: float) -> Tuple[Optional[float], int]:
        """Process an ACK. Returns (rtt_sample or None, number of newly acked chunks).

        Karn's algorithm: an RTT sample is produced only when the ACKed packet
        was never retransmitted, because otherwise we cannot tell which copy
        the ACK belongs to.
        """
        sample = None
        newly = 0
        # Selective part: the packet that triggered this ACK.
        if seq < self.next_seq and not self.acked[seq]:
            entry = self.inflight.get(seq)
            if entry is not None and entry.retx == 0:
                sample = now - entry.first_sent
            if entry is not None and self.rack:
                self._rack_update(entry, now)
            self._mark_acked(seq)
            newly += 1
        # Cumulative part: everything below cum_ack has arrived.
        # Clamp to next_seq: a receiver cannot have data we never sent.
        upper = min(cum_ack, self.next_seq)
        s = self.base
        while s < upper:
            if not self.acked[s]:
                entry = self.inflight.get(s)
                if entry is not None and self.rack:
                    self._rack_update(entry, now)
                self._mark_acked(s)
                newly += 1
            s += 1
        while self.base < self.next_seq and self.acked[self.base]:
            self.base += 1
        return sample, newly

    def _mark_acked(self, seq: int) -> None:
        self.acked[seq] = 1
        self.inflight.pop(seq, None)

    # -- RACK (RFC 8985) ---------------------------------------------------
    def _rack_update(self, entry: InFlight, now: float) -> None:
        rtt = now - entry.last_sent
        if entry.retx and self.min_rtt is not None and rtt < self.min_rtt:
            # Too fast to be the resent copy: the original arrived, only reordered.
            if entry.rack_resent:
                self.spurious_detected += 1
                self.reo_mult += 1
                self._repairs_since_spurious = 0
            return      # ambiguous sample: ignore (RFC 8985 6.2)
        self.min_rtt = rtt if self.min_rtt is None else min(self.min_rtt, rtt)
        if self.rack_xmit is None or entry.last_sent >= self.rack_xmit:
            self.rack_xmit = entry.last_sent
            self.rack_rtt = rtt

    def _rack_deadline(self, entry: InFlight) -> Optional[float]:
        if self.rack_xmit is None or entry.fast_done or entry.last_sent >= self.rack_xmit:
            return None            # nothing sent after it has been delivered yet
        reo_wnd = min(self.reo_mult * self.reo_wnd_frac * self.min_rtt, self.rack_rtt)
        return entry.last_sent + self.rack_rtt + reo_wnd

    def detect_losses(self, now: float) -> List[InFlight]:
        """Packets RACK now considers lost, oldest first; each reported once per transmission."""
        if not self.rack:
            return []
        lost = []
        for e in self.inflight.values():
            dl = self._rack_deadline(e)
            if dl is not None and now >= dl:
                e.fast_done = True
                lost.append(e)
        if lost:
            self._repairs_since_spurious += 1
            if self._repairs_since_spurious >= 16 and self.reo_mult > 1:
                self.reo_mult = 1                 # no recent mistakes: back to the default
                self._repairs_since_spurious = 0
        lost.sort(key=lambda e: e.seq)
        return lost

    def next_rack_deadline(self) -> Optional[float]:
        """When the next packet would be declared lost if nothing else arrives (reorder timer)."""
        if not self.rack:
            return None
        dls = [d for d in (self._rack_deadline(e) for e in self.inflight.values()) if d is not None]
        return min(dls) if dls else None

    def done(self) -> bool:
        return self.base >= self.total


class ReceiveBuffer:
    """Receiver side of Selective Repeat.

    Accepts chunks in [expected, expected + window). In-order chunks are
    released immediately; out-of-order ones are held (at most `window` of them,
    so memory use is bounded no matter what the network sends).
    """

    IN_ORDER = "in_order"
    BUFFERED = "buffered"
    DUPLICATE = "duplicate"
    OUT_OF_WINDOW = "out_of_window"
    INVALID = "invalid"

    def __init__(self, total_chunks: int, window: int):
        self.total = total_chunks
        self.window = window
        self.expected = 0                   # next chunk needed for in-order delivery
        self.buffer: Dict[int, bytes] = {}

    def offer(self, seq: int, data: bytes) -> Tuple[str, List[bytes]]:
        """Returns (status, chunks ready to write in order)."""
        if seq >= self.total:
            return self.INVALID, []
        if seq < self.expected or seq in self.buffer:
            return self.DUPLICATE, []
        if seq >= self.expected + self.window:
            return self.OUT_OF_WINDOW, []
        self.buffer[seq] = data
        ready = []
        while self.expected in self.buffer:
            ready.append(self.buffer.pop(self.expected))
            self.expected += 1
        return (self.IN_ORDER if ready else self.BUFFERED), ready

    def complete(self) -> bool:
        return self.expected >= self.total
