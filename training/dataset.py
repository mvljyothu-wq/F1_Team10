"""Turn raw sender logs into (features, target) rows -- without leakage.

One row = one decision point: the moment an RTT sample arrives at time t and
the timeout is recomputed. Its features were computed by the sender at time t
from past observations only (they are read straight from the log).

Target for that row (see docs/DESIGN.md section 7.1):

    y(t) = 95th percentile of the RTTs of packets SENT in (t, t + H],  H = 0.5 s

i.e. "how long must a timer armed now wait so that ~95% of packets that are
delivered do not time out". Only valid (Karn-filtered) samples exist in the
log, so lost packets do not contribute -- they have no RTT. Rows with fewer
than MIN_FUTURE future samples are dropped (a percentile of 2 numbers is noise).

The future is used ONLY to build the label. Features never look ahead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from protocol.reliability import SenderConfig
from timeout.features import FEATURE_NAMES

HORIZON = 0.5
QUANTILE = 95
MIN_FUTURE = 5
RTO_MIN = SenderConfig.rto_min        # the policy bounds must match the runtime defaults
RTO_MAX = SenderConfig.rto_max
BAND = (0.5, 2.0)


@dataclass
class RunData:
    run: str
    profile: str
    seed: int
    split: str
    rows: pd.DataFrame           # features + t + rto_conv + target + lo/hi indices
    future_rtts: np.ndarray      # RTTs of this run sorted by send time (rows index into it)


def build_run(path: str, run: str, profile: str, seed: int, split: str) -> Optional[RunData]:
    df = pd.read_csv(path)
    s = df[df["event"] == "sample"].dropna(subset=FEATURE_NAMES + ["send_time", "rtt"])
    if len(s) < 2 * MIN_FUTURE:
        return None
    order = np.argsort(s["send_time"].to_numpy(), kind="stable")
    send_sorted = s["send_time"].to_numpy()[order]
    rtt_sorted = s["rtt"].to_numpy()[order]

    t = s["t"].to_numpy()
    lo = np.searchsorted(send_sorted, t, side="right")             # sent strictly after t
    hi = np.searchsorted(send_sorted, t + HORIZON, side="right")   # ... and up to t + H
    target = np.full(len(s), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= MIN_FUTURE:
            target[i] = np.percentile(rtt_sorted[a:b], QUANTILE)

    rows = s[FEATURE_NAMES + ["t", "rto_conv", "rtt"]].copy()
    rows["target"] = target
    rows["lo"], rows["hi"] = lo, hi
    rows = rows[np.isfinite(rows["target"])].reset_index(drop=True)
    rows["run"], rows["profile"], rows["seed"], rows["split"] = run, profile, seed, split
    return RunData(run, profile, seed, split, rows, rtt_sorted)


def load_runs(data_dir: str, splits: Sequence[str], profiles: Optional[Sequence[str]] = None,
              exclude_profiles: Sequence[str] = (), index_files: Sequence[str] = ("runs.csv",)) -> List[RunData]:
    """Load runs from one or more collection rounds (runs.csv = round 1)."""
    index = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in index_files], ignore_index=True)
    index = index[index["split"].isin(splits) & (index["success"] == 1)]
    if profiles:
        index = index[index["profile"].isin(profiles)]
    if exclude_profiles:
        index = index[~index["profile"].isin(exclude_profiles)]
    runs = []
    for r in index.itertuples():
        rd = build_run(os.path.join(data_dir, r.obs_log), r.run, r.profile, int(r.seed), r.split)
        if rd is not None:
            runs.append(rd)
    return runs


def stack(runs: List[RunData]) -> pd.DataFrame:
    return pd.concat([r.rows for r in runs], ignore_index=True)


# -- evaluation helpers (vectorised versions of the runtime logic) ------------

def recent_max_rtt(rows: pd.DataFrame) -> np.ndarray:
    """Largest of the last 16 RTTs, recovered from the features (max_rel * SRTT)."""
    return rows["max_rel"].to_numpy() * np.exp(rows["log_srtt"].to_numpy())


def proposals_from_prediction(pred: np.ndarray, rows: pd.DataFrame, target: str) -> np.ndarray:
    pred = np.clip(pred, -8.0, 8.0)                    # same clip as the runtime estimator
    if target == "correction":
        return rows["rto_conv"].to_numpy() * np.exp(pred)
    return np.exp(pred)


def apply_policy(proposal: np.ndarray, rows: pd.DataFrame, band=BAND,
                 rto_min: float = RTO_MIN, rto_max: float = RTO_MAX) -> np.ndarray:
    """Vectorised timeout.ml_estimator.apply_safety (checked equal in train_model.py)."""
    conv = rows["rto_conv"].to_numpy()
    lo = np.maximum.reduce([np.full_like(conv, rto_min), band[0] * conv, recent_max_rtt(rows)])
    hi = np.minimum(rto_max, band[1] * conv)
    hi = np.maximum(hi, lo)
    return np.minimum(np.clip(proposal, lo, hi), rto_max)


def spurious_fraction_per_row(rows: pd.DataFrame, rto: np.ndarray,
                              futures: Dict[str, np.ndarray]) -> np.ndarray:
    """For each row: fraction of future delivered packets whose RTT exceeds the timeout.
    Those packets would have been retransmitted although their ACK was on the way."""
    out = np.empty(len(rows))
    runs = rows["run"].to_numpy()
    lo = rows["lo"].to_numpy()
    hi = rows["hi"].to_numpy()
    for i in range(len(rows)):
        fut = futures[runs[i]][lo[i]:hi[i]]
        out[i] = np.mean(fut > rto[i])
    return out


def summarize(rows: pd.DataFrame, rto: np.ndarray, futures: Dict[str, np.ndarray]) -> dict:
    y = rows["target"].to_numpy()
    err = rto - y
    spurious = spurious_fraction_per_row(rows, rto, futures)
    over = err[err > 0]
    return {
        "n": int(len(rows)),
        "mae_ms": float(np.mean(np.abs(err)) * 1000),
        "median_ae_ms": float(np.median(np.abs(err)) * 1000),
        "under_target_rate": float(np.mean(err < 0)),
        "spurious_fraction": float(np.mean(spurious)),
        "mean_excess_ms": float(np.mean(over) * 1000) if len(over) else 0.0,
        "mean_ratio": float(np.mean(rto / y)),
    }
