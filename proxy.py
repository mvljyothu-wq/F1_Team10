#!/usr/bin/env python3
"""Run the network-impairment proxy between a sender and a receiver.

Example (receiver on 9000, sender talks to 9001):
    python3 proxy.py --listen-port 9001 --target-port 9000 --profile lossy --seed 1 --status
"""

import argparse
import signal
import sys
import threading

from simulation.profiles import PROFILES, get_profile
from simulation.proxy_core import Proxy


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="UDP network impairment proxy")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=9001)
    ap.add_argument("--target-host", default="127.0.0.1")
    ap.add_argument("--target-port", type=int, default=9000)
    ap.add_argument("--profile", default="lossy", help="network profile (see --list-profiles)")
    ap.add_argument("--seed", type=int, default=1, help="random seed (recorded in stats)")
    ap.add_argument("--stats-file", help="write ground-truth statistics (JSON) here on exit")
    ap.add_argument("--status", action="store_true", help="show a live status line")
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args(argv)

    if args.list_profiles:
        for name, p in PROFILES.items():
            print(f"{name:14s} {p.description}")
        return 0

    proxy = Proxy((args.listen_host, args.listen_port), (args.target_host, args.target_port),
                  get_profile(args.profile), args.seed, status=args.status)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print(f"READY {proxy.port}", flush=True)
    try:
        proxy.run(stop)
    except KeyboardInterrupt:
        pass
    if args.stats_file:
        proxy.write_stats(args.stats_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
