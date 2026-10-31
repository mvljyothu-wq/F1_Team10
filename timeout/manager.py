"""TimeoutManager: the single object the sender asks "how long should I wait?".

It owns
  * the conventional estimator (always updated, in both modes),
  * the observation history and features,
  * the optional ML estimator (mode "ml"),
  * exponential backoff, applied identically in both modes.

Backoff rule (per-packet timers need care): with many packets in flight, one
network outage makes many timers expire together. Doubling the RTO for each of
them would jump straight to rto_max. Instead the manager keeps a "backoff
epoch" number that changes whenever the backoff level changes. Each packet
remembers the epoch in which it was (re)sent, and a timeout raises the level
only if that packet was sent in the CURRENT epoch. So the RTO doubles once per
round of timeouts, like TCP's single timer, and packets sent under an older,
already-reset level cannot push it up again. Any valid (Karn-filtered) RTT
sample resets the level to 0 (RFC 6298 section 5).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from timeout.conventional import ConventionalRTO
from timeout.features import History, compute_features
from timeout.ml_estimator import MLTimeoutEstimator

MAX_BACKOFF_LEVEL = 10


class TimeoutManager:
    def __init__(self, mode: str = "rto", rto_min: float = 0.03, rto_max: float = 10.0,
                 rto_initial: float = 1.0, ml: Optional[MLTimeoutEstimator] = None,
                 ml_interval: float = 0.01, log_features: bool = False, k: float = 4.0):
        if mode not in ("rto", "ml"):
            raise ValueError("mode must be 'rto' or 'ml'")
        self.mode = mode
        self.rto_min = rto_min
        self.rto_max = rto_max
        self.conv = ConventionalRTO(rto_min, rto_max, rto_initial, k=k)
        self.history = History()
        self.ml = ml
        self.ml_interval = ml_interval      # recompute the ML decision at most this often
        self.log_features = log_features

        self.backoff_level = 0
        self.epoch = 0                      # changes whenever backoff_level changes
        self.base_rto = self.conv.rto       # selected RTO before backoff
        self.source = "conv"                # where base_rto came from: conv / ml / fallback
        self.ml_proposal: Optional[float] = None
        self.last_features: Optional[List[float]] = None
        self._last_ml_time = -math.inf

        # counters for reporting
        self.timeouts = 0
        self.backoff_events = 0
        self.ml_decisions = 0
        self.fallbacks: Counter = Counter()

    # -- events from the sender ---------------------------------------------
    def on_transmission(self, is_retx: bool) -> None:
        self.history.add_transmission(is_retx)

    def on_rtt_sample(self, now: float, rtt: float) -> bool:
        """A valid RTT sample (never from a retransmitted packet).
        Returns True if this sample ended a backoff (the caller should re-arm timers)."""
        self.conv.update(rtt)
        self.history.add_sample(now, rtt)
        reset = self.backoff_level != 0
        if reset:
            self.backoff_level = 0
            self.epoch += 1
        self._recompute(now)
        return reset

    def on_timeout(self, now: float, epoch_at_send: int) -> None:
        self.timeouts += 1
        self.history.add_timeout(now)
        if epoch_at_send == self.epoch and self.backoff_level < MAX_BACKOFF_LEVEL:
            self.backoff_level += 1
            self.epoch += 1
            self.backoff_events += 1

    # -- the answer ------------------------------------------------------------
    def current_rto(self) -> float:
        return min(self.rto_max, self.base_rto * (2 ** self.backoff_level))

    # -- internals ---------------------------------------------------------------
    def _features(self, now: float) -> List[float]:
        return compute_features(self.history, self.conv.srtt, self.conv.rttvar, now)

    def _recompute(self, now: float) -> None:
        rto_conv = self.conv.rto
        if self.mode == "rto":
            self.base_rto = rto_conv
            self.source = "conv"
            if self.log_features:
                self.last_features = self._features(now)
            return

        if now - self._last_ml_time < self.ml_interval:
            # Keep the recent ML decision, but if we are currently falling back,
            # follow the conventional value as it updates.
            if self.source != "ml":
                self.base_rto = rto_conv
            if self.log_features:                 # data collection under ML control
                self.last_features = self._features(now)
            return
        self._last_ml_time = now

        features = self._features(now)
        self.last_features = features
        decision = self.ml.decide(features, rto_conv, self.history.total_samples,
                                  self.history.recent_max()) if self.ml else None
        if decision is None:
            self.base_rto, self.source, self.ml_proposal = rto_conv, "fallback", None
            self.fallbacks["no_model"] += 1
            return
        self.base_rto = decision.rto
        self.source = decision.source
        self.ml_proposal = decision.proposal
        if decision.source == "ml":
            self.ml_decisions += 1
        else:
            self.fallbacks[decision.reason] += 1

    def summary(self) -> dict:
        return {
            "timeouts": self.timeouts,
            "backoff_events": self.backoff_events,
            "rtt_samples": self.history.total_samples,
            "srtt": self.conv.srtt,
            "rttvar": self.conv.rttvar,
            "ml_decisions": self.ml_decisions,
            "fallbacks": dict(self.fallbacks),
        }
