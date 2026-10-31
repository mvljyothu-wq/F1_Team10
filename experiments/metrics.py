"""Metric definitions and paired statistics (no SciPy needed).

Per trial (one transfer):
  duration            seconds from first SYN sent to FIN_ACK received
  goodput             file bytes / duration          (useful data rate)
  total traffic rate  all datagram bytes sent by the sender / duration
                      (includes headers, retransmissions, SYN/FIN)
  packets sent        all packets sent by the sender
  retransmissions     DATA packets sent again after a timeout
  timeouts            timer expiries (DATA, SYN, FIN)
  unnecessary retx    retransmissions of a chunk whose earlier copy had already
                      reached the receiver intact (proxy ground truth). Caused by
                      premature timeouts or by lost ACKs.
  loss recovery delay time between a lost DATA copy passing the proxy and the
                      next copy of that chunk (proxy ground truth) = how long
                      the timer made the sender wait before repairing a loss
  packet loss rate    data copies dropped / data copies sent (proxy)
  integrity           SHA-256 verified by the receiver AND recomputed by the harness
  CPU                 sender process CPU seconds (time.process_time)

Paired comparison per profile: d_i = ML_i - conventional_i for the same seed.
  mean difference with a 95% bootstrap confidence interval, and an exact
  two-sided Wilcoxon signed-rank test (small n, no normality assumption).
"""

from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks (1-based) with ties sharing the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_exact(diffs: Sequence[float]) -> Optional[float]:
    """Exact two-sided Wilcoxon signed-rank p-value. Zero differences are dropped.
    Enumerates all 2^n sign assignments (fine for n <= 20)."""
    d = [x for x in diffs if x != 0 and not math.isnan(x)]
    n = len(d)
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError("use a normal approximation for n > 20")
    ranks = _ranks([abs(x) for x in d])
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = sum(ranks)
    expected = total / 2
    observed = abs(w_plus - expected)
    extreme = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum(r for r, s in zip(ranks, signs) if s)
        if abs(w - expected) >= observed - 1e-12:
            extreme += 1
    return extreme / 2 ** n


def bootstrap_ci(diffs: Sequence[float], n_boot: int = 10000, seed: int = 0,
                 level: float = 0.95) -> Tuple[float, float]:
    d = np.asarray([x for x in diffs if not math.isnan(x)], dtype=float)
    if len(d) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired(conv: Sequence[float], ml: Sequence[float]) -> dict:
    conv = np.asarray(conv, dtype=float)
    ml = np.asarray(ml, dtype=float)
    d = ml - conv
    lo, hi = bootstrap_ci(d)
    base = float(np.nanmean(conv))
    return {
        "n": int(len(d)),
        "conv_mean": base, "conv_std": float(np.nanstd(conv, ddof=1)) if len(d) > 1 else 0.0,
        "ml_mean": float(np.nanmean(ml)), "ml_std": float(np.nanstd(ml, ddof=1)) if len(d) > 1 else 0.0,
        "diff_mean": float(np.nanmean(d)), "diff_ci_lo": lo, "diff_ci_hi": hi,
        "pct_change": float(100 * np.nanmean(d) / base) if base else math.nan,
        "p_value": wilcoxon_exact(list(d)),
    }
