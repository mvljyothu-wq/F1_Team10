"""Collect every number the website shows into docs/site/data.json.

Run from the project root after the experiments:
    python3 docs/site/build_data.py && python3 docs/site/build.py

Sources: results/trials.csv, results/summary.csv, results/offline_eval*.csv/json,
results/calibration_v2.csv, results/traces/*.csv, models/*.json, data/raw (one
RTT trace), simulation/profiles.py and the test suite (for the test count).
Nothing is typed in by hand.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from protocol.packet import MsgType, Packet, encode  # noqa: E402
from simulation.profiles import EXPERIMENT_PROFILES, PROFILES  # noqa: E402

ARMS = ["rto", "ml", "rto_k2", "ml_v2", "rto_rack", "ml_v2_rack"]
R = lambda v, d=4: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), d)


def results():
    d = pd.read_csv("results/trials.csv")
    if "fast_retransmits" not in d:                     # results from before RACK existed
        d["fast_retransmits"] = 0
    d["goodput"] = d.goodput_Bps / 1e6
    d["rec"] = d.loss_recovery_mean * 1000
    out, loss = {}, {}
    for (p, m), g in d.groupby(["profile", "mode"]):
        out.setdefault(p, {})[m] = {
            "goodput": R(g.goodput.mean()), "goodput_sd": R(g.goodput.std()),
            "unnec": R(g.unnecessary_retx.mean(), 2), "unnec_sd": R(g.unnecessary_retx.std(), 2),
            "retx": R(g.retransmissions.mean(), 2), "fast": R(g.fast_retransmits.mean(), 2),
            "timeouts": R(g.timeouts.mean(), 2), "rec": R(g.rec.mean(), 1), "dur": R(g.duration.mean()),
            "cpu": R((g.cpu_seconds / g.duration).mean()), "n": int(len(g)),
            "ok": int(g.success.sum()), "verified": int(g.verified.sum())}
        loss.setdefault(p, {})[m] = R(100 * (g.data_copies_lost / g.data_packets_sent).mean(), 2)
    # averages across the 6 lossy networks (stable has no losses), relative to RFC, paired by seed
    avg = {}
    wide = d.pivot_table(index=["profile", "seed"], columns="mode", values=["goodput", "unnecessary_retx"])
    lossy = wide[wide.index.get_level_values(0) != "stable"]
    for m in ARMS:
        if m in d["mode"].unique() and m != "rto":
            g = 100 * (lossy["goodput"][m] - lossy["goodput"]["rto"]) / lossy["goodput"]["rto"]
            w = lossy["unnecessary_retx"][m].groupby(level=0).mean().sum() / \
                lossy["unnecessary_retx"]["rto"].groupby(level=0).mean().sum()
            avg[m] = {"goodput_pct": R(g.mean(), 1), "waste_ratio": R(w, 2)}
    totals = {"transfers": int(len(d)), "ok": int(d.success.sum()), "verified": int(d.verified.sum()),
              "seeds": int(d.seed.nunique()), "arms": sorted(d["mode"].unique().tolist())}
    return out, loss, avg, totals


def paired():
    s = pd.read_csv("results/summary.csv")
    pv = {}
    for r in s.itertuples():
        pv.setdefault(r.comparison, {}).setdefault(r.profile, {})[r.metric] = [
            None if pd.isna(r.pct_change) else round(r.pct_change, 1), round(r.p_value, 4)]
    return pv


def offline(path_csv):
    if not os.path.exists(path_csv):
        return None
    off = pd.read_csv(path_csv)
    return {p: {r.estimator: [round(r.mae_ms, 2), round(r.spurious_fraction, 4)] for r in g.itertuples()}
            for p, g in off.groupby("profile")}


def traces(max_samples=320):
    """Recorded runs (seed 1000) for the live-lab replay, downsampled."""
    out = {}
    trials = pd.read_csv("results/trials.csv")
    if "fast_retransmits" not in trials:
        trials["fast_retransmits"] = 0
    for p in EXPERIMENT_PROFILES:
        for m in ARMS:
            f = f"results/traces/{p}_s1000_{m}.csv"
            if not os.path.exists(f):
                continue
            t = pd.read_csv(f)
            s = t[t.event == "sample"]
            idx = np.unique(np.linspace(0, len(s) - 1, min(max_samples, len(s))).astype(int))
            ss = s.iloc[idx]
            row = trials[(trials.profile == p) & (trials["mode"] == m) & (trials.seed == 1000)].iloc[0]
            out.setdefault(p, {})[m] = {
                "s": [[round(a, 3), round(b * 1000, 1), round(c * 1000, 1), round(e * 1000, 1)]
                      for a, b, c, e in zip(ss.t, ss.rtt, ss.rto_conv, ss.rto_final)],
                "to": [round(x, 3) for x in t[t.event == "timeout"].t],
                "fast": [round(x, 3) for x in t[t.event == "fast"].t] if "fast" in set(t.event) else [],
                "n_samples": int(len(s)),
                "result": {"duration": round(float(row.duration), 3), "goodput": round(row.goodput_Bps / 1e6, 3),
                           "retx": int(row.retransmissions), "fast": int(row.fast_retransmits),
                           "timeouts": int(row.timeouts), "unnec": int(row.unnecessary_retx),
                           "verified": bool(row.verified)}}
    return out


def model_meta(path):
    with open(path) as f:
        doc = json.load(f)
    m = doc.get("metadata", {})
    return {"ensemble": doc.get("ensemble", "forest"), "trees": len(doc["trees"]), "margin": doc["margin"],
            "importances": m.get("feature_importances"), "calibration": m.get("calibration"),
            "candidates": [[c["variant"], c["config"], round(c["val"]["mae_ms"], 2),
                            round(c["val"]["spurious_fraction"], 4)] for c in m.get("candidates", [])],
            "quantile": m.get("quantile_level"), "rounds": m.get("training_rounds"),
            "train_rows": m.get("train_rows"), "size_kb": round(os.path.getsize(path) / 1024)}


def profiles():
    out = {}
    for k in EXPERIMENT_PROFILES:
        p = PROFILES[k]
        out[k] = {"description": p.description, "cycle": p.cycle if p.cycle < 3600 else None,
                  "phases": [dataclasses.asdict(ph) for ph in p.phases],
                  "burst": dataclasses.asdict(p.burst) if p.burst else None}
    return out


def count_tests():
    suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    def n(s):
        return sum(n(x) for x in s) if isinstance(s, unittest.TestSuite) else 1
    return n(suite)


def main():
    res, loss, avg, totals = results()
    oj = json.load(open("results/offline_eval.json"))
    rtt = pd.read_csv("results/traces/jitter_heavy_s1000_rto.csv")
    rtt = rtt[rtt.event == "sample"].rtt.mul(1000).round(2).tolist()
    data = {
        "res": res, "loss": loss, "avg": avg, "totals": totals, "pv": paired(),
        "off": offline("results/offline_eval.csv"), "off_v2": offline("results/offline_eval_v2.csv"),
        "offline_rows": oj["test_rows"], "offline_runs": oj["test_runs"],
        "lopo": {k: [round(v["mae_ms"], 1), round(v["spurious_fraction"], 4)]
                 for k, v in oj.get("lopo", {}).items() if isinstance(v, dict)},
        "v1": model_meta("models/rto_model.json"),
        "v2": model_meta("models/rto_model_v2.json") if os.path.exists("models/rto_model_v2.json") else None,
        "traces": traces(), "profiles": profiles(), "rtt_jitter": rtt,
        "hex": encode(Packet(MsgType.DATA, 0x3A7F12C9, seq=117, ack=0, payload=b"Hello")).hex(),
        "tests": count_tests(),
    }
    out = os.path.join(ROOT, "docs", "site", "data.json")
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB), {data['tests']} tests, "
          f"{totals['transfers']} transfers, arms {totals['arms']}")


if __name__ == "__main__":
    main()
