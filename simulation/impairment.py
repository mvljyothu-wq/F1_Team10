"""Deterministic impairment decisions.

Key idea (needed for fair comparisons): the fate of a packet copy is a pure
function of (seed, direction, message type, sequence number, copy number,
current profile phase). It does NOT depend on how many other packets were sent
before. So if both timeout strategies send "the 3rd copy of chunk 117", that
copy is dropped (or delayed by the same amount) in both runs.

The random numbers come from BLAKE2b of the packet key, not from a shared
random stream. The burst-loss schedule is generated up front from the seed,
so it is a function of time only.

For cyclic (time-varying) profiles each seed also starts at its own offset
within the cycle, so short transfers with different seeds see different parts
of the cycle -- while both timeout strategies run with the same seed still see
exactly the same schedule.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from dataclasses import dataclass
from typing import List, Tuple

from simulation.profiles import Profile

FWD, REV = 0, 1
PARETO_ALPHA = 2.0
PARETO_CAP = 40.0       # cap heavy-tail jitter at 40 x jitter_ms


@dataclass
class Decision:
    drop: bool
    delays: List[float]       # one entry per delivered copy (seconds); empty if dropped
    corrupt: bool
    corrupt_pos: float        # where to flip a byte, as a fraction of the length
    reordered: bool
    duplicated: bool
    in_burst: bool


class Impairment:
    def __init__(self, profile: Profile, seed: int, horizon: float = 3600.0):
        self.profile = profile
        self.seed = seed & 0xFFFFFFFFFFFFFFFF
        self.offset = 0.0
        if profile.cycle < 3600.0:
            self.offset = random.Random(f"offset-{seed}").random() * profile.cycle
        self.bad_periods: List[Tuple[float, float]] = []
        if profile.burst is not None:
            rng = random.Random(f"burst-{seed}")
            t = rng.expovariate(1.0 / profile.burst.mean_good_s)
            while t < horizon:
                bad = rng.expovariate(1.0 / profile.burst.mean_bad_s)
                self.bad_periods.append((t, t + bad))
                t += bad + rng.expovariate(1.0 / profile.burst.mean_good_s)

    def conditions(self, t: float):
        """(phase, one-way base delay ms, in_burst) at transfer time t."""
        phase, delay_ms = self.profile.phase_at(t + self.offset)
        return phase, delay_ms, self.in_burst(t)

    def in_burst(self, t: float) -> bool:
        # Binary search for the first bad period that ends after t.
        lo, hi = 0, len(self.bad_periods)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.bad_periods[mid][1] <= t:
                lo = mid + 1
            else:
                hi = mid
        return lo < len(self.bad_periods) and self.bad_periods[lo][0] <= t

    def _uniforms(self, direction: int, msg_type: int, seq: int, copy: int) -> List[float]:
        key = struct.pack("!QBBII", self.seed, direction, msg_type & 0xFF, seq & 0xFFFFFFFF,
                          copy & 0xFFFFFFFF)
        digest = hashlib.blake2b(key, digest_size=32).digest()
        return [x / 4294967296.0 for x in struct.unpack("!8I", digest)]

    @staticmethod
    def _jitter(kind: str, jitter_ms: float, u1: float, u2: float) -> float:
        if jitter_ms <= 0:
            return 0.0
        if kind == "uniform":
            return (2 * u1 - 1) * jitter_ms
        if kind == "normal":
            z = math.sqrt(-2 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
            return z * jitter_ms
        if kind == "pareto":
            # Pareto(alpha) minus 1: mostly small, occasionally very large.
            return min(PARETO_CAP, (max(u1, 1e-12) ** (-1.0 / PARETO_ALPHA)) - 1.0) * jitter_ms
        raise ValueError(f"unknown jitter kind {kind!r}")

    def decide(self, direction: int, msg_type: int, seq: int, copy: int, t: float) -> Decision:
        u = self._uniforms(direction, msg_type, seq, copy)
        phase, delay_ms, burst = self.conditions(t)
        loss = phase.loss_fwd if direction == FWD else phase.loss_rev
        if burst:
            loss = max(loss, self.profile.burst.bad_loss)
        if u[0] < loss:
            return Decision(True, [], False, 0.0, False, False, burst)

        d_ms = delay_ms + self._jitter(phase.jitter_kind, phase.jitter_ms, u[1], u[2])
        if direction == REV and phase.ack_delay_ms > 0:
            d_ms += u[3] * phase.ack_delay_ms
        reordered = u[4] < phase.reorder
        if reordered:
            d_ms += phase.reorder_ms
        delays = [max(0.0, d_ms) / 1000.0]
        duplicated = u[5] < phase.dup
        if duplicated:
            delays.append(delays[0] + (1.0 + 4.0 * u[3]) / 1000.0)   # second copy 1-5 ms later
        corrupt = u[6] < phase.corrupt
        return Decision(False, delays, corrupt, u[7], reordered, duplicated, burst)
