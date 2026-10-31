"""Feature extraction for the ML timeout estimator.

The SAME function is used when logging training data and when predicting at
runtime, so the two can never drift apart. Every feature uses only information
the sender already has at decision time (no future data, no simulator truth).

Most features are divided by SRTT ("relative"), so the model learns the
*shape* of recent RTT behaviour rather than absolute speeds; one feature
(log_srtt) keeps the absolute scale.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

N_RTT = 16      # recent RTT samples used for statistics
M_TX = 64       # recent transmissions used for the retransmission ratio

FEATURE_NAMES = [
    "log_srtt",          # log(SRTT in seconds): absolute scale
    "rttvar_rel",        # RTTVAR / SRTT: smoothed variability
    "last_rel",          # newest RTT / SRTT
    "mean_rel",          # mean of last N RTTs / SRTT
    "std_rel",           # std of last N RTTs / SRTT
    "p90_rel",           # 90th percentile of last N RTTs / SRTT (tail behaviour)
    "max_rel",           # max of last N RTTs / SRTT
    "min_rel",           # min of last N RTTs / SRTT
    "slope",             # least-squares d(RTT)/d(time) over last N samples (unitless)
    "retx_ratio",        # retransmissions among the last M transmissions
    "since_timeout_rel", # time since last timeout / SRTT (capped)
    "n_samples",         # how much history we have (capped at N)
]

SINCE_TIMEOUT_CAP = 1000.0
SLOPE_CAP = 5.0


class History:
    """Sliding record of what the sender has observed."""

    def __init__(self):
        self.samples: Deque[Tuple[float, float]] = deque(maxlen=N_RTT)   # (time, rtt)
        self.tx: Deque[int] = deque(maxlen=M_TX)                         # 1 = retransmission
        self.last_timeout: Optional[float] = None
        self.total_samples = 0

    def add_sample(self, now: float, rtt: float) -> None:
        self.samples.append((now, rtt))
        self.total_samples += 1

    def add_transmission(self, is_retx: bool) -> None:
        self.tx.append(1 if is_retx else 0)

    def add_timeout(self, now: float) -> None:
        self.last_timeout = now

    def recent_max(self) -> float:
        return max(r for _, r in self.samples) if self.samples else 0.0


def percentile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolation percentile (same definition as numpy's default)."""
    if not sorted_values:
        raise ValueError("empty input")
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _slope(points: List[Tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    mean_t = sum(t for t, _ in points) / n
    mean_r = sum(r for _, r in points) / n
    var_t = sum((t - mean_t) ** 2 for t, _ in points)
    if var_t < 1e-12:
        return 0.0
    cov = sum((t - mean_t) * (r - mean_r) for t, r in points)
    return max(-SLOPE_CAP, min(SLOPE_CAP, cov / var_t))


def compute_features(history: History, srtt: float, rttvar: float, now: float) -> List[float]:
    """Return the feature vector in FEATURE_NAMES order. Requires >= 1 sample."""
    if not history.samples or srtt is None or srtt <= 0:
        raise ValueError("need at least one RTT sample")
    rtts = [r for _, r in history.samples]
    n = len(rtts)
    srt = sorted(rtts)
    mean = sum(rtts) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rtts) / n)
    retx_ratio = (sum(history.tx) / len(history.tx)) if history.tx else 0.0
    if history.last_timeout is None:
        since_to = SINCE_TIMEOUT_CAP
    else:
        since_to = min(SINCE_TIMEOUT_CAP, max(0.0, (now - history.last_timeout) / srtt))
    return [
        math.log(srtt),
        rttvar / srtt,
        rtts[-1] / srtt,
        mean / srtt,
        std / srtt,
        percentile(srt, 90) / srtt,
        srt[-1] / srtt,
        srt[0] / srtt,
        _slope(list(history.samples)),
        retx_ratio,
        since_to,
        float(min(history.total_samples, N_RTT)),
    ]
