"""Network profiles for the impairment proxy.

A profile is a list of phases that repeats cyclically. Each phase sets the
one-way delay (optionally ramping), jitter, loss in each direction,
duplication, reordering, corruption and extra ACK-path delay. A profile may
also have a burst-loss process (Gilbert-Elliott style good/bad periods).

"fwd" = sender -> receiver (SYN, DATA, FIN); "rev" = receiver -> sender (ACKs).
All delays are ONE-WAY, so RTT ~ 2 x delay.

These are application-level simulations: they model delay/loss statistics,
not queueing, bandwidth limits or real router behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Phase:
    duration: float                 # seconds this phase lasts within one cycle
    delay_ms: float                 # base one-way delay (at phase start)
    delay_end_ms: Optional[float] = None   # if set, delay ramps linearly to this
    jitter_ms: float = 0.0
    jitter_kind: str = "uniform"    # uniform: +-jitter; normal: sd=jitter; pareto: heavy tail
    loss_fwd: float = 0.0
    loss_rev: float = 0.0
    dup: float = 0.0                # probability a packet is delivered twice
    reorder: float = 0.0            # probability a packet gets extra delay (overtaken)
    reorder_ms: float = 0.0
    corrupt: float = 0.0            # probability one byte is flipped (fwd and rev)
    ack_delay_ms: float = 0.0       # extra uniform [0, x] delay on the reverse path


@dataclass(frozen=True)
class Burst:
    """Bad periods with high loss, alternating with good periods.
    Durations are exponential; the schedule depends only on the seed."""
    bad_loss: float
    mean_bad_s: float
    mean_good_s: float


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    phases: List[Phase]
    burst: Optional[Burst] = None

    @property
    def cycle(self) -> float:
        return sum(p.duration for p in self.phases)

    def phase_at(self, t: float):
        """Return (phase, current one-way base delay in ms) at elapsed time t."""
        t = t % self.cycle if self.cycle > 0 else 0.0
        for p in self.phases:
            if t < p.duration:
                if p.delay_end_ms is None:
                    return p, p.delay_ms
                frac = t / p.duration
                return p, p.delay_ms + (p.delay_end_ms - p.delay_ms) * frac
            t -= p.duration
        last = self.phases[-1]
        return last, (last.delay_end_ms if last.delay_end_ms is not None else last.delay_ms)


LONG = 3600.0   # a single phase that effectively never ends

PROFILES: Dict[str, Profile] = {p.name: p for p in [
    Profile("perfect", "no impairment (loopback only)", [Phase(LONG, 0.0)]),
    Profile("stable", "low latency, tiny jitter, no loss",
            [Phase(LONG, 10.0, jitter_ms=1.0)]),
    Profile("high_latency", "satellite-like 300 ms RTT, light loss",
            [Phase(LONG, 150.0, jitter_ms=5.0, loss_fwd=0.005, loss_rev=0.005)]),
    Profile("lossy", "5% random data loss, 2% ACK loss",
            [Phase(LONG, 25.0, jitter_ms=2.5, loss_fwd=0.05, loss_rev=0.02)]),
    Profile("bursty_loss", "short bad periods where half the packets are lost",
            [Phase(LONG, 25.0, jitter_ms=2.5, loss_fwd=0.005, loss_rev=0.005)],
            burst=Burst(bad_loss=0.5, mean_bad_s=0.05, mean_good_s=0.8)),
    Profile("jitter_heavy", "heavy-tailed (Pareto) delay spikes, 1% loss",
            [Phase(LONG, 30.0, jitter_ms=4.0, jitter_kind="pareto",
                   loss_fwd=0.01, loss_rev=0.01)]),
    Profile("changing", "path quality changes: ramps and steps every few seconds",
            [Phase(1.5, 15.0, jitter_ms=2.0, loss_fwd=0.01, loss_rev=0.01),
             Phase(1.0, 15.0, delay_end_ms=120.0, jitter_ms=5.0, loss_fwd=0.02, loss_rev=0.01),
             Phase(1.5, 120.0, jitter_ms=10.0, loss_fwd=0.03, loss_rev=0.02),
             Phase(1.0, 30.0, jitter_ms=3.0, loss_fwd=0.01, loss_rev=0.01)]),
    Profile("dup_reorder", "duplication, reordering, corruption and delayed ACKs",
            [Phase(LONG, 20.0, jitter_ms=5.0, loss_fwd=0.01, loss_rev=0.01, dup=0.02,
                   reorder=0.05, reorder_ms=15.0, corrupt=0.01, ack_delay_ms=20.0)]),
]}

EXPERIMENT_PROFILES = ["stable", "high_latency", "lossy", "bursty_loss",
                       "jitter_heavy", "changing", "dup_reorder"]


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from None
