"""Generate training/validation/test traces for the ML timeout estimator.

Every run is a real transfer through the impairment proxy using the
CONVENTIONAL estimator, with the sender logging each RTT sample and the
feature vector computed at that moment (timeout/features.py).

Seeds are split into disjoint ranges BEFORE anything is trained, so no run
can appear in two splits:
    train 0..N-1, validation 100..100+M-1, test 200..200+K-1
The end-to-end experiments (experiments/run_experiments.py) use 1000+ seeds,
which are never seen during training either.

Round 2 (for model v2): the same seeds, but with the sender controlled by the
v1 ML model, so the data also covers the situations the ML estimator itself
creates (more retransmissions, thinner history). This is the idea behind
DAgger-style iterative training and removes the covariate shift seen in v1.

Usage:
    python3 -m training.generate_dataset            # all profiles, default sizes
    python3 -m training.generate_dataset --mode ml --tag mlctl --test-seeds 0   # round 2
    python3 -m training.generate_dataset --train-seeds 4 --val-seeds 2 --test-seeds 2   # quick
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from experiments.harness import ROOT, Lab, make_payload
from simulation.profiles import EXPERIMENT_PROFILES

SPLIT_BASE = {"train": 0, "val": 100, "test": 200}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", nargs="+", default=EXPERIMENT_PROFILES)
    ap.add_argument("--train-seeds", type=int, default=20)
    ap.add_argument("--val-seeds", type=int, default=6)
    ap.add_argument("--test-seeds", type=int, default=6)
    ap.add_argument("--file-size", type=int, default=1 << 20, help="bytes per transfer (default 1 MiB)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--ports", type=int, nargs=2, default=[9600, 9601], metavar=("RECEIVER", "PROXY"))
    ap.add_argument("--mode", choices=["rto", "ml"], default="rto", help="timeout mode while collecting")
    ap.add_argument("--model", default=str(ROOT / "models" / "rto_model.json"), help="model for --mode ml")
    ap.add_argument("--tag", default="", help="name for this collection round (default: none = round 1)")
    args = ap.parse_args(argv)

    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    runs_path = os.path.join(args.data_dir, f"runs_{args.tag}.csv" if args.tag else "runs.csv")
    prefix = f"{args.tag}_" if args.tag else ""
    plan = []
    for split, n in (("train", args.train_seeds), ("val", args.val_seeds), ("test", args.test_seeds)):
        plan += [(split, SPLIT_BASE[split] + i) for i in range(n)]

    t_start = time.time()
    with Lab(os.path.join(args.data_dir, "work"), *args.ports) as lab, \
            open(runs_path, "w", newline="") as runs_file:
        w = csv.writer(runs_file)
        w.writerow(["run", "profile", "seed", "split", "success", "verified", "duration",
                    "retransmissions", "rtt_samples", "unnecessary_retx", "obs_log"])
        total = len(plan) * len(args.profiles)
        done = 0
        for profile in args.profiles:
            for split, seed in plan:
                run = f"{prefix}{profile}_s{seed}"
                src = make_payload(os.path.join(args.data_dir, "work", "payloads", f"seed{seed}.bin"),
                                   args.file_size, seed)
                log_path = os.path.join(raw_dir, f"{run}.csv")
                res, stats, verified = lab.run_trial(src, profile, seed, args.mode, obs_log=log_path,
                                                     log_features=True, model_path=args.model)
                w.writerow([run, profile, seed, split, int(res.success), int(verified),
                            f"{res.duration:.4f}", res.retransmissions, res.rtt_samples,
                            stats["unnecessary_retransmissions"], os.path.relpath(log_path, args.data_dir)])
                runs_file.flush()
                done += 1
                print(f"[{done:4d}/{total}] {run:22s} {split:5s} ok={res.success} "
                      f"{res.duration:6.2f}s samples={res.rtt_samples}  "
                      f"elapsed {time.time() - t_start:6.0f}s", flush=True)
    print(f"wrote {runs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
