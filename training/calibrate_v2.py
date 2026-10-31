"""End-to-end calibration of the v2 model (the step v1 was missing).

v1 chose its safety margin offline and turned out about twice as wasteful
online as RFC 6298. v2 chooses its quantile level and margin by running real
transfers on VALIDATION seeds (100-102), with a rule fixed before looking at
any result:

    1. For every candidate (quantile level x margin) and for RFC 6298, run the
       same validation transfers: 7 networks x 3 seeds, same impairment seeds.
    2. A candidate qualifies if its total unnecessary retransmissions are at
       most 1.15 x those of RFC 6298 on the same transfers.
    3. Among qualifying candidates, pick the highest mean paired goodput gain
       over RFC 6298. If none qualifies, pick the least wasteful candidate.

The result is written to models/rto_model_v2.json together with the full
calibration table. Test and experiment seeds (200+, 1000+) are never used here.

Usage:
    python3 -m training.calibrate_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time

import numpy as np

from experiments.harness import ROOT, Lab, make_payload
from simulation.profiles import EXPERIMENT_PROFILES

SEEDS = [100, 101, 102]
MARGINS = [1.0, 1.15, 1.3]
WASTE_LIMIT = 1.15


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", default=str(ROOT / "models" / "v2_candidates"))
    ap.add_argument("--out", default=str(ROOT / "models" / "rto_model_v2.json"))
    ap.add_argument("--work", default=str(ROOT / "data" / "work_calibration"))
    ap.add_argument("--ports", type=int, nargs=2, default=[9620, 9621])
    args = ap.parse_args(argv)

    cands = sorted(f for f in os.listdir(args.candidates) if f.endswith(".json"))
    configs = [("rfc", None, None)] + [(c[:-5], os.path.join(args.candidates, c), m)
                                        for c in cands for m in MARGINS]
    rows = []
    t0 = time.time()
    total = len(configs) * len(EXPERIMENT_PROFILES) * len(SEEDS)
    with Lab(args.work, *args.ports) as lab:
        for profile in EXPERIMENT_PROFILES:
            for seed in SEEDS:
                src = make_payload(os.path.join(args.work, "payloads", f"seed{seed}.bin"), 1 << 20, seed)
                for name, path, margin in configs:
                    if path is None:
                        res, st, ok = lab.run_trial(src, profile, seed, "rto")
                    else:
                        res, st, ok = lab.run_trial(src, profile, seed, "ml", model_path=path, ml_margin=margin)
                    rows.append({"config": name if path is None else f"{name}_m{margin}", "cand": name,
                                 "margin": margin, "profile": profile, "seed": seed,
                                 "ok": int(res.success and ok), "goodput": res.goodput_Bps / 1e6,
                                 "unnec": st["unnecessary_retransmissions"]})
                    print(f"[{len(rows):4d}/{total}] {profile:13s} s{seed} {rows[-1]['config']:12s} "
                          f"{rows[-1]['goodput']:.3f} MB/s unnec={rows[-1]['unnec']:4d}  "
                          f"{time.time() - t0:5.0f}s", flush=True)

    os.makedirs(ROOT / "results", exist_ok=True)
    with open(ROOT / "results" / "calibration_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    base = {(r["profile"], r["seed"]): r for r in rows if r["config"] == "rfc"}
    base_waste = sum(r["unnec"] for r in base.values())
    table = []
    for name, path, margin in configs[1:]:
        cfg = f"{name}_m{margin}"
        sub = [r for r in rows if r["config"] == cfg]
        gains = [100 * (r["goodput"] - base[(r["profile"], r["seed"])]["goodput"]) /
                 base[(r["profile"], r["seed"])]["goodput"] for r in sub]
        waste = sum(r["unnec"] for r in sub)
        table.append({"config": cfg, "path": path, "margin": margin, "quantile": name,
                      "mean_goodput_gain_pct": round(float(np.mean(gains)), 2),
                      "waste_ratio_vs_rfc": round(waste / max(base_waste, 1), 3),
                      "all_ok": all(r["ok"] for r in sub)})
    ok = [t for t in table if t["waste_ratio_vs_rfc"] <= WASTE_LIMIT and t["all_ok"]]
    best = max(ok, key=lambda t: t["mean_goodput_gain_pct"]) if ok else min(table, key=lambda t: t["waste_ratio_vs_rfc"])
    print("\ncandidate      gain vs RFC   waste vs RFC")
    for t in table:
        mark = " <- chosen" if t is best else ""
        print(f"  {t['config']:12s} {t['mean_goodput_gain_pct']:+7.2f} %   x{t['waste_ratio_vs_rfc']:.2f}{mark}")

    with open(best["path"]) as f:
        doc = json.load(f)
    doc["margin"] = best["margin"]
    doc["metadata"]["calibration"] = {
        "rule": f"waste <= {WASTE_LIMIT} x RFC on validation transfers, then max goodput gain",
        "seeds": SEEDS, "profiles": EXPERIMENT_PROFILES, "rfc_unnecessary_total": base_waste,
        "table": [{k: v for k, v in t.items() if k != "path"} for t in table],
        "chosen": best["config"], "qualified": bool(ok)}
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    shutil.rmtree(os.path.join(args.work, "received"), ignore_errors=True)
    print(f"wrote {args.out} ({best['config']}); table in results/calibration_v2.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
