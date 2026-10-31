"""Conventional retransmission timeout estimator (RFC 6298, Jacobson/Karels).

    first sample R:   SRTT = R,  RTTVAR = R/2
    later samples R:  RTTVAR = (1-beta)*RTTVAR + beta*|SRTT - R|   (uses the OLD SRTT)
                      SRTT   = (1-alpha)*SRTT + alpha*R
    RTO = clamp(SRTT + max(G, K*RTTVAR), rto_min, rto_max)

alpha = 1/8, beta = 1/4, K = 4 are the RFC values. G is the clock granularity.

This class only computes the *base* RTO. Exponential backoff and Karn's rule
are applied by the caller (timeout/manager.py and protocol/window.py), so the
same backoff applies to both timeout strategies.
"""

from __future__ import annotations

import math
from typing import Optional


class ConventionalRTO:
    def __init__(self, rto_min: float = 0.03, rto_max: float = 10.0, rto_initial: float = 1.0,
                 alpha: float = 0.125, beta: float = 0.25, k: float = 4.0,
                 granularity: float = 0.001):
        if not 0 < rto_min <= rto_max:
            raise ValueError("need 0 < rto_min <= rto_max")
        self.rto_min = rto_min
        self.rto_max = rto_max
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.granularity = granularity
        self.srtt: Optional[float] = None
        self.rttvar: Optional[float] = None
        self.samples = 0
        self.rto = self._clamp(rto_initial)   # used until the first sample

    def _clamp(self, value: float) -> float:
        return min(max(value, self.rto_min), self.rto_max)

    def update(self, rtt: float) -> float:
        """Feed one valid (Karn-filtered) RTT sample in seconds; returns new RTO."""
        if not math.isfinite(rtt) or rtt < 0:
            raise ValueError(f"invalid RTT sample {rtt!r}")
        if self.srtt is None:
            self.srtt = rtt
            self.rttvar = rtt / 2
        else:
            self.rttvar = (1 - self.beta) * self.rttvar + self.beta * abs(self.srtt - rtt)
            self.srtt = (1 - self.alpha) * self.srtt + self.alpha * rtt
        self.samples += 1
        self.rto = self._clamp(self.srtt + max(self.granularity, self.k * self.rttvar))
        return self.rto
