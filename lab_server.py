#!/usr/bin/env python3
"""Live lab: serve the project website and run real transfers from the browser.

    python3 lab_server.py            # then open http://localhost:8000

The page starts a transfer with POST /api/run, then listens on
GET /api/events?run=<id>, a Server-Sent Events stream that carries every RTT
sample, timeout and fast retransmit as it happens, progress snapshots, and the
final result with the proxy's ground truth. Everything is a real transfer:
a receiver process and an impairment-proxy process on this machine, and the
real sender running in this server.

Only the Python standard library is used. The server listens on 127.0.0.1
only, accepts one transfer at a time, validates every input against a fixed
list, and serves files only from docs/site/.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from experiments.harness import Lab, make_payload
from simulation.profiles import PROFILES

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "docs" / "site"
MODES = {
    "rto": {"mode": "rto"},
    "ml": {"mode": "ml", "model_path": str(ROOT / "models" / "rto_model.json")},
    "ml_v2": {"mode": "ml", "model_path": str(ROOT / "models" / "rto_model_v2.json")},
}
SIZES_KB = [256, 512, 1024, 2048, 4096]
MAX_BODY = 4096
KEEP_RUNS = 10


class Run:
    """One transfer and its event log (kept so a late subscriber can replay it)."""

    def __init__(self, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.params = params
        self.events: List[str] = []
        self.done = False
        self.cond = threading.Condition()

    def emit(self, kind: str, data: dict) -> None:
        line = f"event: {kind}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        with self.cond:
            self.events.append(line)
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify_all()


class LabRunner:
    """Owns the receiver process and runs one transfer at a time."""

    def __init__(self, workdir: Path, ports=(9800, 9801)):
        self.lab = Lab(str(workdir), *ports)
        self.workdir = workdir
        self.lock = threading.Lock()
        self.busy = False
        self.runs: Dict[str, Run] = {}

    def start(self) -> None:
        self.lab.__enter__()

    def stop(self) -> None:
        self.lab.__exit__(None, None, None)

    def submit(self, params: dict) -> Run:
        with self.lock:
            if self.busy:
                raise RuntimeError("busy")
            self.busy = True
            run = Run(params)
            self.runs[run.id] = run
            for old in list(self.runs)[:-KEEP_RUNS]:
                del self.runs[old]
        threading.Thread(target=self._execute, args=(run,), daemon=True).start()
        return run

    def _execute(self, run: Run) -> None:
        p = run.params
        try:
            src = make_payload(str(self.workdir / "payloads" / f"live_{p['size_kb']}k_s{p['seed']}.bin"),
                               p["size_kb"] * 1024, p["seed"])
            spec = dict(MODES[p["mode"]])
            mode = spec.pop("mode")
            run.emit("start", {"params": p, "profile": PROFILES[p["profile"]].description})
            last = [0.0]

            def progress(s):
                now = time.monotonic()
                if now - last[0] >= 0.1 or s["phase"] == "done":
                    last[0] = now
                    run.emit("progress", {k: s[k] for k in (
                        "phase", "elapsed", "chunks_acked", "total_chunks", "acked_bytes", "file_size",
                        "throughput_Bps", "packets_sent", "retransmissions", "fast_retransmits",
                        "timeouts", "in_flight", "srtt", "rto_final", "source", "backoff_level")})

            def events(kind, d):
                run.emit(kind, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()})

            res, stats, verified = self.lab.run_trial(
                src, p["profile"], p["seed"], mode, fast_retransmit=p["rack"],
                progress_interval=0.1, progress=progress, events=events, **spec)
            run.emit("done", {"result": res.to_dict(), "proxy": stats, "verified": verified})
        except Exception as exc:                          # report, never crash the server
            run.emit("failed", {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            run.finish()
            with self.lock:
                self.busy = False


def parse_params(body: dict) -> dict:
    """Validate a run request against fixed lists; raises ValueError."""
    profile = body.get("profile", "lossy")
    mode = body.get("mode", "rto")
    size = int(body.get("size_kb", 1024))
    seed = int(body.get("seed", 7))
    rack = bool(body.get("rack", False))
    if profile not in PROFILES:
        raise ValueError("unknown profile")
    if mode not in MODES:
        raise ValueError("unknown mode")
    if mode != "rto" and not os.path.exists(MODES[mode]["model_path"]):
        raise ValueError(f"model for {mode} not found; train it first")
    if size not in SIZES_KB:
        raise ValueError(f"size_kb must be one of {SIZES_KB}")
    if not 0 <= seed < 100000:
        raise ValueError("seed out of range")
    return {"profile": profile, "mode": mode, "size_kb": size, "seed": seed, "rack": rack}


def make_handler(runner: LabRunner):
    class Handler(BaseHTTPRequestHandler):
        server_version = "udp-rto-lab/1"

        def log_message(self, fmt, *args):       # keep the terminal quiet
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/api/health":
                return self._json(200, {"ok": True, "busy": runner.busy, "profiles": sorted(PROFILES),
                                        "modes": [m for m in MODES if m == "rto" or os.path.exists(MODES[m]["model_path"])],
                                        "sizes_kb": SIZES_KB})
            if url.path == "/api/events":
                return self._events(parse_qs(url.query).get("run", [""])[0])
            return self._static(url.path)

        def do_POST(self):
            if urlparse(self.path).path != "/api/run":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_BODY:
                    raise ValueError("request too large")
                params = parse_params(json.loads(self.rfile.read(length) or b"{}"))
                run = runner.submit(params)
            except RuntimeError:
                return self._json(409, {"error": "a transfer is already running"})
            except (ValueError, json.JSONDecodeError) as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(200, {"run": run.id, "params": params})

        def _events(self, run_id: str):
            run = runner.runs.get(run_id)
            if run is None:
                return self._json(404, {"error": "unknown run"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sent = 0
            try:
                while True:
                    with run.cond:
                        while sent == len(run.events) and not run.done:
                            run.cond.wait(timeout=15)
                            if sent == len(run.events) and not run.done:
                                self.wfile.write(b": keep-alive\n\n")
                                self.wfile.flush()
                        batch = run.events[sent:]
                        finished = run.done
                    if batch:
                        self.wfile.write("".join(batch).encode())
                        self.wfile.flush()
                        sent += len(batch)
                    if finished and sent == len(run.events):
                        return
            except (BrokenPipeError, ConnectionResetError):
                return

        def _static(self, path: str):
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (SITE / rel).resolve()
            if SITE.resolve() not in target.parents and target != SITE.resolve() or not target.is_file():
                return self._json(404, {"error": "not found"})
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(port: int, stop: Optional[threading.Event] = None, ready=None, lab_ports=(9800, 9801)) -> None:
    runner = LabRunner(ROOT / "data" / "work_live", lab_ports)
    runner.start()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(runner))
    httpd.daemon_threads = True
    if ready:
        ready(httpd.server_address[1])
    try:
        if stop is None:
            httpd.serve_forever()
        else:
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            stop.wait()
    finally:
        httpd.shutdown()
        httpd.server_close()
        runner.stop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the project website with the live lab")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    print(f"Live lab running: open http://localhost:{args.port}   (Ctrl+C to stop)", flush=True)
    try:
        serve(args.port)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
