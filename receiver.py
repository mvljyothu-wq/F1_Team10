#!/usr/bin/env python3
"""Receive files over the reliable UDP protocol.

Example:
    python3 receiver.py --host 0.0.0.0 --port 9000 --output-dir received_files
"""

import argparse
import logging
import signal
import sys
import threading

from protocol.reliability import Receiver, ReceiverConfig, read_key_file


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reliable-UDP file receiver")
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=9000, help="UDP port (default 9000)")
    ap.add_argument("--output-dir", default="received_files", help="where files are written")
    ap.add_argument("--max-file-size", type=int, default=1 << 30, help="bytes (default 1 GiB)")
    ap.add_argument("--max-window", type=int, default=1024, help="largest sender window accepted")
    ap.add_argument("--idle-timeout", type=float, default=30.0, help="drop silent transfers (s)")
    ap.add_argument("--key-file", help="file with a pre-shared HMAC key (enables authentication)")
    ap.add_argument("--once", action="store_true", help="exit after one transfer finishes")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    key = read_key_file(args.key_file) if args.key_file else None
    receiver = Receiver(ReceiverConfig(host=args.host, port=args.port, output_dir=args.output_dir,
                                       max_file_size=args.max_file_size, max_window=args.max_window,
                                       idle_timeout=args.idle_timeout, key=key))
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    logging.info("listening on %s:%d, writing to %s%s", args.host, receiver.port,
                 receiver.output_dir, " (HMAC enabled)" if key else "")
    print(f"READY {receiver.port}", flush=True)       # used by scripts to know we are up
    try:
        receiver.serve(stop, once=args.once)
    except KeyboardInterrupt:
        pass
    ok = sum(1 for _, s in receiver.completed if s == "OK")
    logging.info("stopped: %d transfer(s) completed successfully, %d failed", ok,
                 len(receiver.completed) - ok)
    return 0 if (not args.once or ok == len(receiver.completed)) else 1


if __name__ == "__main__":
    sys.exit(main())
