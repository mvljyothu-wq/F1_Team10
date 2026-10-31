#!/usr/bin/env python3
"""Send a file over the reliable UDP protocol.

Examples:
    python3 sender.py --file sample.bin --host 127.0.0.1 --port 9000 --timeout-mode rto
    python3 sender.py --file sample.bin --port 9001 --timeout-mode ml --dashboard
"""

import argparse
import json
import logging
import os
import sys

from protocol.packet import MAX_PAYLOAD
from protocol.reliability import SenderConfig, read_key_file, send_file


def _ms(x):
    return f"{x * 1000:7.1f} ms" if x is not None else "      -   "


class Dashboard:
    """A small terminal dashboard redrawn in place (ANSI escape codes)."""

    LINES = 9

    def __init__(self):
        self.drawn = False

    def __call__(self, s: dict) -> None:
        if self.drawn:
            sys.stdout.write(f"\x1b[{self.LINES}F")        # move cursor up to redraw
        self.drawn = True
        frac = s["acked_bytes"] / s["file_size"] if s["file_size"] else 1.0
        bar = "#" * int(frac * 40)
        ml = _ms(s["ml_proposal"]) if s["mode"] == "ml" else "  (rto mode)"
        source = {"conv": "conventional", "ml": "ML", "fallback": "conventional (ML fallback)"}
        lines = [
            f" File        {s['filename']}  [{s['phase']}]  mode={s['mode']}",
            f" Progress    [{bar:<40s}] {frac * 100:5.1f}%  "
            f"{s['chunks_acked']}/{s['total_chunks']} chunks",
            f" Packets     sent {s['packets_sent']:6d}   in flight {s['in_flight']:4d}   "
            f"retransmissions {s['retransmissions']:5d}   timeouts {s['timeouts']:5d}",
            f" RTT         last {_ms(s['last_rtt'])}   smoothed {_ms(s['srtt'])}",
            f" RTO         conventional {_ms(s['rto_conv'])}   ML proposal {ml}",
            f" Selected    {_ms(s['rto_final'])}  from {source.get(s['source'], s['source'])}"
            f"   backoff x{2 ** s['backoff_level']}",
            f" Throughput  {s['throughput_Bps'] / 1e6:6.3f} MB/s (file data acknowledged)",
            f" Elapsed     {s['elapsed']:6.2f} s",
            f" Integrity   {'verifying on receiver...' if s['phase'] == 'finishing' else '-'}",
        ]
        sys.stdout.write("".join(f"\x1b[2K{line}\n" for line in lines))
        sys.stdout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reliable-UDP file sender")
    ap.add_argument("--file", required=True, help="file to send")
    ap.add_argument("--host", default="127.0.0.1", help="receiver (or proxy) address")
    ap.add_argument("--port", type=int, default=9000, help="receiver (or proxy) port")
    ap.add_argument("--timeout-mode", choices=["rto", "ml"], default="rto",
                    help="rto = RFC 6298 estimator, ml = ML estimator with safety bounds")
    ap.add_argument("--model", default="models/rto_model.json", help="ML model (JSON)")
    ap.add_argument("--ml-margin", type=float, help="override the safety margin stored in the model")
    ap.add_argument("--payload-size", type=int, default=MAX_PAYLOAD, help=f"bytes per packet (max {MAX_PAYLOAD})")
    ap.add_argument("--window", type=int, default=64, help="max packets in flight")
    ap.add_argument("--rto-min", type=float, default=0.03, help="minimum timeout (s)")
    ap.add_argument("--rto-max", type=float, default=10.0, help="maximum timeout (s)")
    ap.add_argument("--rto-k", type=float, default=4.0, help="K in SRTT + K*RTTVAR (RFC 6298: 4)")
    ap.add_argument("--rto-initial", type=float, default=1.0, help="timeout before any RTT sample (s)")
    ap.add_argument("--fast-retransmit", action="store_true",
                    help="RACK loss detection (RFC 8985): resend without waiting for the timer")
    ap.add_argument("--max-retries", type=int, default=15, help="retransmissions per packet before giving up")
    ap.add_argument("--idle-timeout", type=float, default=30.0, help="abort if receiver silent (s)")
    ap.add_argument("--obs-log", help="write RTT samples and timeouts to this CSV")
    ap.add_argument("--log-features", action="store_true", help="include ML features in --obs-log")
    ap.add_argument("--key-file", help="file with a pre-shared HMAC key")
    ap.add_argument("--json-out", help="write the transfer result as JSON")
    ap.add_argument("--dashboard", action="store_true", help="live terminal dashboard")
    ap.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(message)s")
    if not os.path.isfile(args.file):
        print(f"error: {args.file!r} is not a readable file", file=sys.stderr)
        return 2
    cfg = SenderConfig(host=args.host, port=args.port, payload_size=args.payload_size,
                       window=args.window, timeout_mode=args.timeout_mode, model_path=args.model,
                       ml_margin=args.ml_margin, rto_min=args.rto_min, rto_max=args.rto_max,
                       rto_initial=args.rto_initial, rto_k=args.rto_k, fast_retransmit=args.fast_retransmit, max_retries=args.max_retries,
                       idle_timeout=args.idle_timeout, obs_log=args.obs_log,
                       log_features=args.log_features,
                       key=read_key_file(args.key_file) if args.key_file else None)
    result = send_file(args.file, cfg, progress=Dashboard() if args.dashboard else None)

    r = result
    print()
    print(f"Result      : {'SUCCESS' if r.success else 'FAILED'} ({r.status})"
          + (f" - {r.error}" if r.error else ""))
    print(f"Integrity   : {'SHA-256 verified by receiver' if r.success else 'NOT verified'}")
    print(f"Size / time : {r.file_size} bytes in {r.duration:.3f} s")
    print(f"Goodput     : {r.goodput_Bps / 1e6:.3f} MB/s (file data)   "
          f"total traffic {r.total_Bps / 1e6:.3f} MB/s")
    print(f"Packets     : {r.packets_sent} sent, {r.data_packets_sent} data, "
          f"{r.retransmissions} retransmissions ({r.fast_retransmits} fast), {r.timeouts} timeouts")
    if r.rtt_samples:
        print(f"RTT         : mean {r.rtt_mean * 1000:.1f} ms, min {r.rtt_min * 1000:.1f}, "
              f"max {r.rtt_max * 1000:.1f} ({r.rtt_samples} samples)")
    if r.timeout_mode == "ml":
        print(f"ML          : {r.ml_decisions} decisions, fallbacks {r.ml_fallbacks}"
              + (f", model error: {r.model_error}" if r.model_error else ""))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(r.to_dict(), f, indent=2)
    return 0 if r.success else 1


if __name__ == "__main__":
    sys.exit(main())
