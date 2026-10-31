#!/usr/bin/env python3
"""One-command demonstration: receiver + impairment proxy + sender with live dashboard.

Examples:
    python3 demo.py                                  # 2 MB sample, lossy network, conventional RTO
    python3 demo.py --profile changing --timeout-mode ml
    python3 demo.py --profile jitter_heavy --compare # run both modes on the same seed, then compare
    python3 demo.py --file report.pdf --profile perfect

The receiver and proxy run as background processes and are stopped at the end.
Received files go to demo_out/ (deleted before each run).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from experiments.harness import ROOT, _start, _stop, make_payload
from protocol.reliability import SenderConfig, send_file
from sender import Dashboard
from simulation.profiles import PROFILES

RECV_PORT, PROXY_PORT = 9400, 9401


def run_once(src, args, mode):
    out = ROOT / "demo_out"
    shutil.rmtree(out, ignore_errors=True)
    receiver, _ = _start(["receiver.py", "--host", "127.0.0.1", "--port", str(RECV_PORT),
                          "--output-dir", str(out), "--log-level", "WARNING"], ROOT)
    proxy, _ = _start(["proxy.py", "--listen-port", str(PROXY_PORT), "--target-port", str(RECV_PORT),
                       "--profile", args.profile, "--seed", str(args.seed)], ROOT)
    try:
        print(f"\n=== {mode.upper()} mode | profile '{args.profile}' "
              f"({PROFILES[args.profile].description}) | seed {args.seed} ===\n")
        res = send_file(src, SenderConfig(port=PROXY_PORT, timeout_mode=mode, model_path=args.model),
                        progress=Dashboard())
    finally:
        time.sleep(0.1)
        _stop(proxy)
        _stop(receiver)
    print(f"\n  {'SUCCESS' if res.success else 'FAILED'} ({res.status})"
          + (f": {res.error}" if res.error else "") +
          f" - {res.file_size:,} bytes in {res.duration:.2f} s, goodput {res.goodput_Bps / 1e6:.2f} MB/s, "
          f"{res.retransmissions} retransmissions, {res.timeouts} timeouts")
    if mode == "ml":
        print(f"  ML decisions {res.ml_decisions}, fallbacks {res.ml_fallbacks}"
              + (f", model error: {res.model_error}" if res.model_error else ""))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live demo of the reliable UDP transfer")
    ap.add_argument("--file", help="file to send (default: generate a sample)")
    ap.add_argument("--size-mb", type=float, default=2.0, help="size of the generated sample")
    ap.add_argument("--profile", default="lossy", choices=sorted(PROFILES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout-mode", default="rto", choices=["rto", "ml"])
    ap.add_argument("--model", default=str(ROOT / "models" / "rto_model.json"))
    ap.add_argument("--compare", action="store_true", help="run rto and ml on the same seed")
    args = ap.parse_args(argv)

    src = args.file or make_payload(str(ROOT / "demo_out_sample.bin"), int(args.size_mb * 1e6), 1)
    modes = ["rto", "ml"] if args.compare else [args.timeout_mode]
    results = {m: run_once(src, args, m) for m in modes}
    if args.compare:
        a, b = results["rto"], results["ml"]
        print("\n=== Same network seed, two timeout strategies ===")
        print(f"  {'':22s}{'conventional':>14s}{'ML':>14s}")
        for label, fa, fb in [("transfer time (s)", a.duration, b.duration),
                              ("goodput (MB/s)", a.goodput_Bps / 1e6, b.goodput_Bps / 1e6),
                              ("retransmissions", a.retransmissions, b.retransmissions),
                              ("timeouts", a.timeouts, b.timeouts)]:
            print(f"  {label:22s}{fa:14.3f}{fb:14.3f}" if isinstance(fa, float)
                  else f"  {label:22s}{fa:14d}{fb:14d}")
        print("  (one seed is an anecdote: see results/RESULTS.md for repeated trials)")
    return 0 if all(r.success for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
