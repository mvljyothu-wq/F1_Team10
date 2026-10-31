"""Paired end-to-end experiments: conventional vs ML timeout on identical conditions.

For every (profile, seed) the same file is sent twice -- once per timeout mode --
through the proxy with the same seed. Because impairment decisions are keyed by
(seed, packet, copy number) and not by a shared random stream, both runs face
the same network fate for the same packet copies (see simulation/impairment.py).

The order of the two modes alternates between seeds (rto first on even seeds,
ml first on odd), so slow drift in machine load cannot systematically favour
one mode. Seeds start at 1000 and never overlap the training/validation/test
seeds used for the model.

Usage:
    python3 -m experiments.run_experiments                  # 7 profiles x 10 seeds x 2 modes
    python3 -m experiments.run_experiments --seeds 3 --profiles lossy changing    # quick
    python3 -m experiments.run_experiments --resume         # continue an interrupted run
    python3 -m experiments.run_experiments --modes rto_k2 --resume   # add the K=2 ablation arm

Modes ("arms"):
  rto         RFC 6298, K=4 (the baseline)
  ml          ML estimator v1 (random forest)
  rto_k2      RFC formula with K=2: a deliberately more aggressive timeout, used to ask
              whether an ML estimator is smarter or merely shorter
  ml_v2       ML estimator v2 (quantile boosting, retrained under ML control,
              calibrated end to end on validation transfers)
  rto_rack    RFC 6298 + RACK fast loss detection (RFC 8985)
  ml_v2_rack  ML v2 + RACK
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from experiments.harness import ROOT, Lab, make_payload
from simulation.profiles import EXPERIMENT_PROFILES

COLUMNS = ["profile", "seed", "mode", "order", "success", "verified", "duration", "goodput_Bps",
           "total_Bps", "bytes_sent", "packets_sent", "data_packets_sent", "retransmissions",
           "fast_retransmits",
           "timeouts", "rtt_mean", "final_srtt", "ml_decisions", "ml_fallbacks", "cpu_seconds",
           "max_rss_bytes", "unnecessary_retx", "data_copies_lost", "loss_recovery_mean",
           "loss_recovery_p95", "fwd_dropped", "rev_dropped", "error"]


V2_MODEL = str(ROOT / "models" / "rto_model_v2.json")
MODES = {"rto": {"mode": "rto"},
         "ml": {"mode": "ml"},
         "rto_k2": {"mode": "rto", "rto_k": 2.0},
         "ml_v2": {"mode": "ml", "model_path": V2_MODEL},
         "rto_rack": {"mode": "rto", "fast_retransmit": True},
         "ml_v2_rack": {"mode": "ml", "model_path": V2_MODEL, "fast_retransmit": True}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Paired conventional-vs-ML transfer experiments")
    ap.add_argument("--profiles", nargs="+", default=EXPERIMENT_PROFILES)
    ap.add_argument("--seeds", type=int, default=10, help="paired trials per profile")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--file-size", type=int, default=1 << 20)
    ap.add_argument("--model", default=str(ROOT / "models" / "rto_model.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--keep-traces", type=int, default=2,
                    help="keep per-sample logs for the first N seeds (for timeline plots)")
    ap.add_argument("--ports", type=int, nargs=2, default=[9500, 9501], metavar=("RECEIVER", "PROXY"))
    ap.add_argument("--resume", action="store_true", help="skip trials already in trials.csv")
    ap.add_argument("--modes", nargs="+", default=["rto", "ml"], choices=sorted(MODES))
    args = ap.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"model not found: {args.model} (run python3 -m training.train_model first)")
        return 1
    os.makedirs(os.path.join(args.out_dir, "traces"), exist_ok=True)
    trials_path = os.path.join(args.out_dir, "trials.csv")
    done = set()
    if args.resume and os.path.exists(trials_path):
        with open(trials_path) as f:
            done = {(r["profile"], int(r["seed"]), r["mode"]) for r in csv.DictReader(f)}
    if done:
        with open(trials_path) as f:
            old = list(csv.DictReader(f))
        if old and list(old[0].keys()) != COLUMNS:      # file from an older version: add new columns
            with open(trials_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                for r in old:
                    w.writerow({c: r.get(c, 0) for c in COLUMNS})
    mode = "a" if done else "w"

    seeds = [args.seed_base + i for i in range(args.seeds)]
    total = len(args.profiles) * len(seeds) * len(args.modes)
    count = 0
    t0 = time.time()
    with Lab(os.path.join(args.out_dir, "work"), *args.ports) as lab, \
            open(trials_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        if mode == "w":
            w.writeheader()
        for profile in args.profiles:
            for i, seed in enumerate(seeds):
                src = make_payload(os.path.join(args.out_dir, "work", "payloads", f"seed{seed}.bin"),
                                   args.file_size, seed)
                # rotate the order of modes from seed to seed
                shift = seed % len(args.modes)
                order = args.modes[shift:] + args.modes[:shift]
                for k, tmode in enumerate(order):
                    count += 1
                    if (profile, seed, tmode) in done:
                        continue
                    trace = (os.path.join(args.out_dir, "traces", f"{profile}_s{seed}_{tmode}.csv")
                             if i < args.keep_traces else None)
                    spec = dict(MODES[tmode])
                    tm_mode = spec.pop("mode")
                    model = spec.pop("model_path", args.model)
                    res, st, verified = lab.run_trial(src, profile, seed, tm_mode,
                                                      model_path=model, obs_log=trace, **spec)
                    w.writerow({
                        "profile": profile, "seed": seed, "mode": tmode, "order": k,
                        "success": int(res.success), "verified": int(verified),
                        "duration": f"{res.duration:.5f}", "goodput_Bps": f"{res.goodput_Bps:.1f}",
                        "total_Bps": f"{res.total_Bps:.1f}", "bytes_sent": res.bytes_sent,
                        "packets_sent": res.packets_sent, "data_packets_sent": res.data_packets_sent,
                        "retransmissions": res.retransmissions, "timeouts": res.timeouts,
                        "fast_retransmits": res.fast_retransmits,
                        "rtt_mean": f"{res.rtt_mean:.6f}" if res.rtt_mean else "",
                        "final_srtt": f"{res.final_srtt:.6f}" if res.final_srtt else "",
                        "ml_decisions": res.ml_decisions,
                        "ml_fallbacks": sum(res.ml_fallbacks.values()),
                        "cpu_seconds": f"{res.cpu_seconds:.4f}", "max_rss_bytes": res.max_rss_bytes,
                        "unnecessary_retx": st["unnecessary_retransmissions"],
                        "data_copies_lost": st["data_copies_lost"],
                        "loss_recovery_mean": st["loss_recovery_mean"] or "",
                        "loss_recovery_p95": st["loss_recovery_p95"] or "",
                        "fwd_dropped": st["counts"]["fwd"]["dropped"],
                        "rev_dropped": st["counts"]["rev"]["dropped"],
                        "error": res.error or "",
                    })
                    fh.flush()
                    print(f"[{count:4d}/{total}] {profile:13s} seed {seed} {tmode:3s} "
                          f"ok={res.success}/{verified} {res.duration:6.2f}s retx={res.retransmissions:4d} "
                          f"unnec={st['unnecessary_retransmissions']:4d}  elapsed {time.time() - t0:5.0f}s",
                          flush=True)
    print(f"wrote {trials_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
