"""Reliable file transfer over UDP: sender and receiver state machines.

Sender flow   : SYN -> (SYN_ACK) -> DATA window with per-packet timers -> FIN -> (FIN_ACK)
Receiver flow : validate SYN -> buffer/write DATA in order, ACK every packet ->
                on FIN verify SHA-256 -> atomically rename -> FIN_ACK(status)

Both sides are single-threaded loops that block on the socket with a timeout
equal to the next deadline (no busy waiting, no locks). See docs/DESIGN.md.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import secrets
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional

from protocol.packet import (
    MAX_DATAGRAM, MAX_PAYLOAD, MAX_WINDOW, U32_MAX, MsgType, Packet, PacketError,
    Status, SynInfo, chunk_length, decode, decode_status, decode_syn, encode,
    encode_status, encode_syn, expected_chunks,
)
from protocol.window import ReceiveBuffer, SendWindow
from timeout.features import FEATURE_NAMES
from timeout.manager import TimeoutManager
from timeout.ml_estimator import load_estimator

log = logging.getLogger("rudp")

RECV_BUFSIZE = MAX_DATAGRAM + 1     # one byte extra so oversized datagrams are detectable


def _max_rss_bytes() -> Optional[int]:
    try:
        import resource
    except ImportError:          # Windows
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024   # macOS: bytes, Linux: KiB


def sha256_file(path: str) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.digest()


# =============================================================================
# Sender
# =============================================================================

@dataclass
class SenderConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    payload_size: int = MAX_PAYLOAD
    window: int = 64
    timeout_mode: str = "rto"                 # "rto" (conventional) or "ml"
    model_path: Optional[str] = "models/rto_model.json"
    ml_margin: Optional[float] = None         # override the margin stored in the model
    rto_min: float = 0.03
    rto_max: float = 10.0
    rto_initial: float = 1.0
    rto_k: float = 4.0                        # K in SRTT + K*RTTVAR (RFC value 4)
    max_retries: int = 15                     # per packet, then the transfer is aborted
    handshake_retries: int = 8                # SYN / FIN attempts
    idle_timeout: float = 30.0                # abort if the receiver is silent this long
    fast_retransmit: bool = False             # RACK-style loss detection (RFC 8985)
    reo_wnd_frac: float = 0.25                # RACK reordering window = this x min RTT
    obs_log: Optional[str] = None             # CSV of RTT samples / timeouts
    log_features: bool = False                # add ML features to obs_log
    key: Optional[bytes] = None               # HMAC key (both ends must match)
    progress_interval: float = 0.25           # seconds between progress callbacks


@dataclass
class TransferResult:
    success: bool
    status: str                       # FIN_ACK status name, or "ERROR"
    error: Optional[str]
    filename: str
    file_size: int
    duration: float                   # SYN sent -> FIN_ACK received (s)
    goodput_Bps: float                # file bytes / duration
    total_Bps: float                  # all datagram bytes sent / duration
    packets_sent: int                 # all packet types
    data_packets_sent: int            # DATA incl. retransmissions
    retransmissions: int              # DATA retransmissions
    timeouts: int                     # timer expiries (DATA + SYN + FIN)
    fast_retransmits: int             # DATA resent by RACK before the timer (included in retransmissions)
    bytes_sent: int                   # UDP payload bytes sent
    acks_received: int
    invalid_received: int             # corrupted / malformed datagrams dropped
    rtt_samples: int
    rtt_mean: Optional[float]
    rtt_min: Optional[float]
    rtt_max: Optional[float]
    final_srtt: Optional[float]
    final_rto: float
    timeout_mode: str
    ml_decisions: int
    ml_fallbacks: Dict[str, int]
    model_error: Optional[str]
    cpu_seconds: float
    max_rss_bytes: Optional[int]
    transfer_id: int

    def to_dict(self) -> dict:
        return asdict(self)


class TransferError(Exception):
    """The transfer could not be completed (never reported as success)."""


ProgressFn = Callable[[dict], None]
EventFn = Callable[[str, dict], None]     # ("sample" | "timeout" | "fast", details)


class _Sender:
    LOG_COLUMNS = ["event", "t", "seq", "send_time", "rtt", "srtt", "rttvar", "rto_conv",
                   "rto_base", "rto_final", "source", "ml_proposal", "backoff_level"]

    def __init__(self, path: str, cfg: SenderConfig, progress: Optional[ProgressFn],
                 events: Optional[EventFn] = None):
        if not 1 <= cfg.payload_size <= MAX_PAYLOAD:
            raise ValueError(f"payload size must be 1..{MAX_PAYLOAD}")
        if not 1 <= cfg.window <= MAX_WINDOW:
            raise ValueError(f"window must be 1..{MAX_WINDOW}")
        self.cfg = cfg
        self.progress = progress
        self.events = events
        self.path = path
        self.filename = os.path.basename(path)
        self.file_size = os.path.getsize(path)
        self.total = expected_chunks(self.file_size, cfg.payload_size)
        if self.total > U32_MAX:
            raise ValueError("file too large for 32-bit sequence numbers")
        self.sha256 = sha256_file(path)
        self.transfer_id = secrets.randbits(32)

        ml, self.model_error = (None, None)
        if cfg.timeout_mode == "ml":
            ml, self.model_error = load_estimator(cfg.model_path, cfg.rto_min, cfg.rto_max,
                                                  cfg.ml_margin)
            if self.model_error:
                log.warning("ML model unavailable (%s); using conventional RTO", self.model_error)
        self.tm = TimeoutManager(cfg.timeout_mode, cfg.rto_min, cfg.rto_max, cfg.rto_initial,
                                 ml=ml, k=cfg.rto_k, log_features=cfg.log_features)

        # counters
        self.packets_sent = self.data_packets_sent = self.retransmissions = 0
        self.fast_retransmits = 0
        self.bytes_sent = self.acks_received = self.invalid_received = 0
        self.rtt_sum, self.rtt_min, self.rtt_max = 0.0, None, None
        self.last_rtt: Optional[float] = None
        self.window: Optional[SendWindow] = None
        self.phase = "init"

        self.sock: Optional[socket.socket] = None
        self.fh = None
        self.log_file = None
        self.log_writer = None
        self.t0 = 0.0
        self.last_rx = 0.0
        self._next_progress = 0.0

    # -- low-level I/O ------------------------------------------------------
    def _send(self, pkt: Packet) -> None:
        raw = encode(pkt, self.cfg.key)
        try:
            self.sock.send(raw)
        except ConnectionRefusedError:
            pass        # ICMP error from an earlier datagram; the packet is simply lost
        self.packets_sent += 1
        self.bytes_sent += len(raw)

    def _recv(self, timeout: float) -> Optional[Packet]:
        self.sock.settimeout(max(timeout, 1e-4))
        try:
            data = self.sock.recv(RECV_BUFSIZE)
        except socket.timeout:
            return None
        except ConnectionRefusedError:
            # Linux reports "port unreachable" on a connected UDP socket this way.
            # Treat it as loss; the timers will retry.
            time.sleep(min(timeout, 0.01))
            return None
        try:
            pkt = decode(data, self.cfg.key)
        except PacketError as exc:
            self.invalid_received += 1
            log.debug("dropped invalid datagram: %s", exc)
            return None
        if pkt.transfer_id != self.transfer_id:
            return None
        self.last_rx = time.monotonic()
        if pkt.msg_type == MsgType.ERROR:
            status, msg = decode_status(pkt.payload) if pkt.payload else (Status.REJECTED, "")
            raise TransferError(f"receiver error {status.name}: {msg}")
        return pkt

    def _send_data(self, seq: int) -> None:
        self.fh.seek(seq * self.cfg.payload_size)
        data = self.fh.read(self.cfg.payload_size)
        self._send(Packet(MsgType.DATA, self.transfer_id, seq=seq, payload=data))
        self.data_packets_sent += 1

    # -- logging / progress -------------------------------------------------
    def _open_log(self) -> None:
        if not self.cfg.obs_log:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cfg.obs_log)), exist_ok=True)
        self.log_file = open(self.cfg.obs_log, "w", newline="")
        self.log_writer = csv.writer(self.log_file)
        cols = list(self.LOG_COLUMNS)
        if self.tm.log_features:
            cols += FEATURE_NAMES
        self.log_writer.writerow(cols)

    def _log_row(self, event: str, now: float, seq: int, rtt: Optional[float]) -> None:
        tm = self.tm
        if self.events is not None:
            try:
                self.events(event, {"t": now - self.t0, "seq": seq, "rtt": rtt,
                                    "srtt": tm.conv.srtt, "rto_conv": tm.conv.rto,
                                    "rto_final": tm.current_rto(), "ml_proposal": tm.ml_proposal,
                                    "source": tm.source, "backoff": tm.backoff_level})
            except Exception as exc:              # a broken observer must not break the transfer
                log.debug("event hook failed: %s", exc)
        if not self.log_writer:
            return
        row = [event, f"{now - self.t0:.6f}", seq,
               f"{now - rtt - self.t0:.6f}" if rtt is not None else "",
               f"{rtt:.6f}" if rtt is not None else "",
               f"{tm.conv.srtt:.6f}" if tm.conv.srtt is not None else "",
               f"{tm.conv.rttvar:.6f}" if tm.conv.rttvar is not None else "",
               f"{tm.conv.rto:.6f}", f"{tm.base_rto:.6f}", f"{tm.current_rto():.6f}",
               tm.source, f"{tm.ml_proposal:.6f}" if tm.ml_proposal is not None else "",
               tm.backoff_level]
        if tm.log_features:
            feats = tm.last_features if (event == "sample" and tm.last_features) else None
            row += [f"{v:.6g}" for v in feats] if feats else [""] * len(FEATURE_NAMES)
        self.log_writer.writerow(row)

    def _on_sample(self, now: float, rtt: float, seq: int) -> None:
        backoff_ended = self.tm.on_rtt_sample(now, rtt)
        if backoff_ended and self.window is not None:
            self.window.rearm_all(self.tm.current_rto(), self.tm.epoch)
        self.last_rtt = rtt
        self.rtt_sum += rtt
        self.rtt_min = rtt if self.rtt_min is None else min(self.rtt_min, rtt)
        self.rtt_max = rtt if self.rtt_max is None else max(self.rtt_max, rtt)
        self._log_row("sample", now, seq, rtt)

    def snapshot(self) -> dict:
        now = time.monotonic()
        elapsed = max(now - self.t0, 1e-9)
        acked = self.window.base if self.window else 0
        acked_bytes = min(acked * self.cfg.payload_size, self.file_size)
        return {
            "phase": self.phase, "elapsed": elapsed, "filename": self.filename,
            "chunks_acked": acked, "total_chunks": self.total, "file_size": self.file_size,
            "acked_bytes": acked_bytes, "throughput_Bps": acked_bytes / elapsed,
            "packets_sent": self.packets_sent, "retransmissions": self.retransmissions,
            "fast_retransmits": self.fast_retransmits, "acks_received": self.acks_received,
            "timeouts": self.tm.timeouts, "in_flight": len(self.window.inflight) if self.window else 0,
            "last_rtt": self.last_rtt, "srtt": self.tm.conv.srtt, "rto_conv": self.tm.conv.rto,
            "ml_proposal": self.tm.ml_proposal, "rto_final": self.tm.current_rto(),
            "source": self.tm.source, "backoff_level": self.tm.backoff_level,
            "mode": self.cfg.timeout_mode, "model_error": self.model_error,
        }

    def _maybe_progress(self, force: bool = False) -> None:
        if self.progress is None:
            return
        now = time.monotonic()
        if force or now >= self._next_progress:
            self._next_progress = now + self.cfg.progress_interval
            self.progress(self.snapshot())

    def _check_idle(self) -> None:
        if time.monotonic() - self.last_rx > self.cfg.idle_timeout:
            raise TransferError(f"no valid reply from receiver for {self.cfg.idle_timeout:.0f}s")

    # -- protocol phases ----------------------------------------------------
    def _exchange(self, pkt: Packet, expect: MsgType) -> Packet:
        """Send a control packet and wait for its reply, retransmitting with backoff."""
        attempt = 0
        while True:
            epoch = self.tm.epoch
            sent_at = time.monotonic()
            self._send(pkt)
            self.tm.on_transmission(attempt > 0)
            deadline = sent_at + self.tm.current_rto()
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                reply = self._recv(min(deadline - now, self.cfg.progress_interval))
                self._maybe_progress()
                if reply is not None and reply.msg_type == expect:
                    if attempt == 0:                      # Karn: only unambiguous samples
                        t = time.monotonic()
                        self._on_sample(t, t - sent_at, 0)
                    return reply
            self.tm.on_timeout(time.monotonic(), epoch)
            self._log_row("timeout", time.monotonic(), 0, None)
            attempt += 1
            if attempt > self.cfg.handshake_retries:
                raise TransferError(f"no {expect.name} after {attempt} attempts "
                                    "(receiver down, wrong address, or HMAC key mismatch?)")

    def _handshake(self) -> None:
        self.phase = "handshake"
        info = SynInfo(self.file_size, self.total, self.cfg.payload_size, self.cfg.window,
                       self.sha256, self.filename)
        self._exchange(Packet(MsgType.SYN, self.transfer_id, payload=encode_syn(info)),
                       MsgType.SYN_ACK)

    def _data_phase(self) -> None:
        self.phase = "data"
        win = SendWindow(self.total, self.cfg.window, rack=self.cfg.fast_retransmit,
                         reo_wnd_frac=self.cfg.reo_wnd_frac)
        self.window = win
        tm = self.tm
        while not win.done():
            # 1. retransmit every packet whose timer has expired
            now = time.monotonic()
            for entry in win.pop_expired(now):
                tm.on_timeout(now, entry.epoch)
                self._log_row("timeout", now, entry.seq, None)
                if entry.retx >= self.cfg.max_retries:
                    raise TransferError(f"chunk {entry.seq} not acknowledged after "
                                        f"{entry.retx} retransmissions")
                self._send_data(entry.seq)
                t = time.monotonic()
                win.on_retransmit(entry.seq, t, t + tm.current_rto(), tm.epoch)
                tm.on_transmission(True)
                self.retransmissions += 1

            # 2. fill the window with new packets
            while win.can_send_new():
                seq = win.next_seq
                self._send_data(seq)
                t = time.monotonic()
                win.on_new_send(t, t + tm.current_rto(), tm.epoch)
                tm.on_transmission(False)

            # 3. wait for an ACK, at most until the earliest timer expires
            self._check_idle()
            nd = win.next_deadline()
            rd = win.next_rack_deadline()
            if rd is not None:
                nd = rd if nd is None else min(nd, rd)
            now = time.monotonic()
            wait = (nd - now) if nd is not None else self.cfg.progress_interval
            pkt = self._recv(min(wait, self.cfg.progress_interval))
            if pkt is not None and pkt.msg_type == MsgType.ACK:
                self.acks_received += 1
                t = time.monotonic()
                sample, _ = win.on_ack(pkt.seq, pkt.ack, t)
                if sample is not None:
                    self._on_sample(t, sample, pkt.seq)
            # 4. fast loss detection (RACK): a packet sent before one that has already
            #    been delivered, and outstanding longer than RTT + reordering window,
            #    is resent now. It is not a timeout, so there is no backoff.
            for entry in win.detect_losses(time.monotonic()):
                if entry.retx >= self.cfg.max_retries:
                    continue                      # leave it to the timer, which aborts cleanly
                self._send_data(entry.seq)
                t = time.monotonic()
                win.on_retransmit(entry.seq, t, t + tm.current_rto(), tm.epoch, by_rack=True)
                tm.on_transmission(True)
                self.retransmissions += 1
                self.fast_retransmits += 1
                self._log_row("fast", t, entry.seq, None)
            self._maybe_progress()

    def _finish(self) -> Status:
        self.phase = "finishing"
        reply = self._exchange(Packet(MsgType.FIN, self.transfer_id), MsgType.FIN_ACK)
        status, msg = decode_status(reply.payload)
        if msg:
            log.info("receiver: %s", msg)
        return status

    # -- entry point --------------------------------------------------------
    def run(self) -> TransferResult:
        cpu0 = time.process_time()
        status_name, error = "ERROR", None
        self.t0 = self.last_rx = time.monotonic()
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # connect() makes the kernel drop datagrams from any other address
            self.sock.connect((self.cfg.host, self.cfg.port))
            self.fh = open(self.path, "rb")
            self._open_log()
            self._handshake()
            self._data_phase()
            status = self._finish()
            status_name = status.name
            if status != Status.OK:
                error = f"receiver reported {status.name}"
        except TransferError as exc:
            error = str(exc)
        except OSError as exc:
            error = f"socket/file error: {exc}"
        finally:
            duration = time.monotonic() - self.t0
            self.phase = "done"
            self._maybe_progress(force=True)
            for closable in (self.fh, self.log_file, self.sock):
                if closable is not None:
                    closable.close()
        n = self.tm.history.total_samples
        summary = self.tm.summary()
        return TransferResult(
            success=(status_name == "OK"), status=status_name, error=error,
            filename=self.filename, file_size=self.file_size, duration=duration,
            goodput_Bps=self.file_size / duration if status_name == "OK" else 0.0,
            total_Bps=self.bytes_sent / duration if duration > 0 else 0.0,
            packets_sent=self.packets_sent, data_packets_sent=self.data_packets_sent,
            retransmissions=self.retransmissions, timeouts=self.tm.timeouts,
            fast_retransmits=self.fast_retransmits,
            bytes_sent=self.bytes_sent, acks_received=self.acks_received,
            invalid_received=self.invalid_received, rtt_samples=n,
            rtt_mean=(self.rtt_sum / n) if n else None, rtt_min=self.rtt_min,
            rtt_max=self.rtt_max, final_srtt=summary["srtt"], final_rto=self.tm.current_rto(),
            timeout_mode=self.cfg.timeout_mode, ml_decisions=summary["ml_decisions"],
            ml_fallbacks=summary["fallbacks"], model_error=self.model_error,
            cpu_seconds=time.process_time() - cpu0, max_rss_bytes=_max_rss_bytes(),
            transfer_id=self.transfer_id)


def send_file(path: str, cfg: SenderConfig, progress: Optional[ProgressFn] = None,
              events: Optional[EventFn] = None) -> TransferResult:
    """Send one file. Never raises for network problems: check result.success.

    progress(snapshot) is called every cfg.progress_interval seconds; events(kind, details)
    is called for every RTT sample, timeout and fast retransmit (used by the live lab)."""
    return _Sender(path, cfg, progress, events).run()


# =============================================================================
# Receiver
# =============================================================================

@dataclass
class ReceiverConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    output_dir: str = "received_files"
    max_file_size: int = 1 << 30          # 1 GiB
    max_window: int = 1024                # largest sender window we accept (buffer bound)
    max_transfers: int = 8                # concurrent transfers
    idle_timeout: float = 30.0            # drop a transfer after this much silence
    linger: float = 5.0                   # keep finished state to re-answer duplicate FINs
    replay_window: float = 600.0          # refuse reuse of a finished transfer ID for this long
    key: Optional[bytes] = None


def safe_filename(name: str) -> str:
    """Validate a filename received from the network. Raises ValueError if unsafe.

    We reject rather than "fix" suspicious names, so nothing surprising is created.
    """
    if not name or len(name.encode("utf-8")) > 255:
        raise ValueError("empty or too long")
    if name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("path components are not allowed")
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("control characters are not allowed")
    return name


def _unique_path(directory: str, name: str, reserved: set) -> str:
    base, ext = os.path.splitext(name)
    candidate = os.path.join(directory, name)
    i = 1
    while os.path.exists(candidate) or candidate in reserved:
        candidate = os.path.join(directory, f"{base} ({i}){ext}")
        i += 1
    return candidate


@dataclass
class _Transfer:
    tid: int
    addr: tuple
    info: SynInfo
    syn_payload: bytes
    final_path: str
    part_path: str
    fh: object
    buf: ReceiveBuffer
    started: float
    last_activity: float
    hasher: object = field(default_factory=hashlib.sha256)
    state: str = "active"                 # active / done
    done_time: float = 0.0
    fin_reply: Optional[bytes] = None
    data_packets: int = 0
    duplicates: int = 0
    out_of_window: int = 0


class Receiver:
    def __init__(self, cfg: ReceiverConfig):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        self.output_dir = os.path.realpath(cfg.output_dir)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((cfg.host, cfg.port))
        self.port = self.sock.getsockname()[1]
        self.transfers: Dict[int, _Transfer] = {}
        self.recently_finished: Dict[int, float] = {}
        self.completed = []                  # list of (final_path or None, Status name)
        self.stats = {"invalid": 0, "unknown_transfer": 0, "wrong_source": 0,
                      "protocol_errors": 0, "rejected": 0}

    # -- sending helpers ----------------------------------------------------
    def _send(self, pkt: Packet, addr) -> None:
        try:
            self.sock.sendto(encode(pkt, self.cfg.key), addr)
        except OSError as exc:
            log.debug("sendto failed: %s", exc)

    def _reject(self, tid: int, addr, status: Status, msg: str) -> None:
        self.stats["rejected"] += 1
        log.warning("rejecting transfer %08x from %s: %s", tid, addr, msg)
        self._send(Packet(MsgType.ERROR, tid, payload=encode_status(status, msg)), addr)

    # -- packet handlers ----------------------------------------------------
    def handle(self, data: bytes, addr, now: float) -> None:
        try:
            pkt = decode(data, self.cfg.key)
        except PacketError as exc:
            self.stats["invalid"] += 1
            log.debug("dropped invalid datagram from %s: %s", addr, exc)
            return
        t = self.transfers.get(pkt.transfer_id)
        if t is not None and t.addr != addr:
            self.stats["wrong_source"] += 1      # a transfer is bound to one source address
            return
        if pkt.msg_type == MsgType.SYN:
            self._on_syn(pkt, addr, now, t)
        elif t is None:
            self.stats["unknown_transfer"] += 1
        elif pkt.msg_type == MsgType.DATA:
            self._on_data(t, pkt, now)
        elif pkt.msg_type == MsgType.FIN:
            self._on_fin(t, pkt, now)
        else:
            self.stats["protocol_errors"] += 1

    def _on_syn(self, pkt: Packet, addr, now: float, t: Optional[_Transfer]) -> None:
        tid = pkt.transfer_id
        if t is not None:
            if t.syn_payload == pkt.payload and t.state == "active":
                self._send(Packet(MsgType.SYN_ACK, tid), addr)    # duplicate SYN: idempotent
            elif t.syn_payload != pkt.payload:
                self._reject(tid, addr, Status.REJECTED, "transfer id already in use")
            return
        if tid in self.recently_finished:
            self._reject(tid, addr, Status.REJECTED, "transfer id recently used (replay?)")
            return
        if sum(1 for x in self.transfers.values() if x.state == "active") >= self.cfg.max_transfers:
            self._reject(tid, addr, Status.REJECTED, "receiver busy")
            return
        try:
            info = decode_syn(pkt.payload)
            name = safe_filename(info.filename)
        except (PacketError, ValueError) as exc:
            self._reject(tid, addr, Status.REJECTED, f"bad SYN: {exc}")
            return
        if info.file_size > self.cfg.max_file_size:
            self._reject(tid, addr, Status.REJECTED, "file too large")
            return
        if info.window > self.cfg.max_window:
            self._reject(tid, addr, Status.REJECTED, f"window > {self.cfg.max_window}")
            return
        # Names of transfers still in progress are reserved (their files do not
        # exist yet). Finished transfers do not reserve anything: their file is
        # either on disk already or was deliberately removed.
        reserved = {x.final_path for x in self.transfers.values() if x.state == "active"}
        final_path = _unique_path(self.output_dir, name, reserved)
        if os.path.dirname(os.path.realpath(final_path)) != self.output_dir:
            self._reject(tid, addr, Status.REJECTED, "unsafe output path")   # defence in depth
            return
        part_path = f"{final_path}.{tid:08x}.part"
        try:
            fh = open(part_path, "xb")       # "x": never overwrite an existing file
        except OSError as exc:
            self._reject(tid, addr, Status.INTERNAL_ERROR, f"cannot create file: {exc}")
            return
        self.transfers[tid] = _Transfer(tid, addr, info, bytes(pkt.payload), final_path,
                                        part_path, fh, ReceiveBuffer(info.total_chunks, info.window),
                                        now, now)
        log.info("transfer %08x: receiving %r (%d bytes, %d chunks, window %d) from %s",
                 tid, name, info.file_size, info.total_chunks, info.window, addr)
        self._send(Packet(MsgType.SYN_ACK, tid), addr)

    def _on_data(self, t: _Transfer, pkt: Packet, now: float) -> None:
        t.last_activity = now
        if t.state != "active":
            # Late duplicate after completion: acknowledge so the sender is never stuck.
            self._send(Packet(MsgType.ACK, t.tid, seq=pkt.seq, ack=t.info.total_chunks), t.addr)
            return
        if pkt.seq >= t.info.total_chunks or len(pkt.payload) != chunk_length(t.info, pkt.seq):
            self.stats["protocol_errors"] += 1
            return
        t.data_packets += 1
        status, ready = t.buf.offer(pkt.seq, pkt.payload)
        if status == ReceiveBuffer.OUT_OF_WINDOW:
            t.out_of_window += 1             # sender is ahead of what we allow: drop, no ACK
            return
        if status == ReceiveBuffer.DUPLICATE:
            t.duplicates += 1                # still ACK it: the previous ACK may have been lost
        for chunk in ready:
            t.fh.write(chunk)
            t.hasher.update(chunk)
        self._send(Packet(MsgType.ACK, t.tid, seq=pkt.seq, ack=t.buf.expected), t.addr)

    def _on_fin(self, t: _Transfer, pkt: Packet, now: float) -> None:
        t.last_activity = now
        if t.fin_reply is None:
            if not t.buf.complete():
                status, msg = Status.INCOMPLETE, f"only {t.buf.expected}/{t.info.total_chunks} chunks"
                self._discard(t)
            else:
                t.fh.flush()
                os.fsync(t.fh.fileno())
                t.fh.close()
                if t.hasher.digest() == t.info.sha256:
                    final = t.final_path
                    if os.path.exists(final):          # appeared since SYN: pick another name
                        final = _unique_path(self.output_dir, os.path.basename(final), set())
                    os.replace(t.part_path, final)
                    t.final_path = final
                    status, msg = Status.OK, "SHA-256 verified"
                else:
                    os.remove(t.part_path)
                    status, msg = Status.HASH_MISMATCH, "SHA-256 mismatch, file discarded"
            t.fin_reply = encode(Packet(MsgType.FIN_ACK, t.tid, payload=encode_status(status, msg)),
                                 self.cfg.key)
            t.state = "done"
            t.done_time = now
            self.recently_finished[t.tid] = now
            self.completed.append((t.final_path if status == Status.OK else None, status.name))
            log.info("transfer %08x: %s (%s) in %.2fs, %d data packets, %d duplicates",
                     t.tid, status.name, msg, now - t.started, t.data_packets, t.duplicates)
        try:
            self.sock.sendto(t.fin_reply, t.addr)       # also re-sent for duplicate FINs
        except OSError:
            pass

    def _discard(self, t: _Transfer) -> None:
        try:
            if not t.fh.closed:
                t.fh.close()
            if os.path.exists(t.part_path):
                os.remove(t.part_path)
        except OSError as exc:
            log.warning("could not remove %s: %s", t.part_path, exc)

    # -- housekeeping -------------------------------------------------------
    def housekeeping(self, now: float) -> None:
        for tid, t in list(self.transfers.items()):
            if t.state == "active" and now - t.last_activity > self.cfg.idle_timeout:
                log.warning("transfer %08x: idle for %.0fs, aborting and deleting partial file",
                            tid, self.cfg.idle_timeout)
                self._discard(t)
                self.completed.append((None, "ABORTED_IDLE"))
                del self.transfers[tid]
            elif t.state == "done" and now - t.done_time > self.cfg.linger:
                del self.transfers[tid]
        for tid, ts in list(self.recently_finished.items()):
            if now - ts > self.cfg.replay_window:
                del self.recently_finished[tid]

    def serve(self, stop: Optional[threading.Event] = None, once: bool = False) -> None:
        """Run until `stop` is set (or, with once=True, after one transfer finishes
        and its linger period has passed)."""
        try:
            while stop is None or not stop.is_set():
                self.sock.settimeout(0.2)
                try:
                    data, addr = self.sock.recvfrom(RECV_BUFSIZE)
                    self.handle(data, addr, time.monotonic())
                except socket.timeout:
                    pass
                except ConnectionResetError:
                    pass
                self.housekeeping(time.monotonic())
                if once and self.completed and not self.transfers:
                    return
        finally:
            for t in self.transfers.values():
                if t.state == "active":
                    self._discard(t)
            self.sock.close()


def read_key_file(path: str) -> bytes:
    """Read a pre-shared HMAC key. Keys are read from a file, never from the
    command line, because command lines are visible to other users (ps)."""
    with open(path, "rb") as f:
        key = f.read().strip()
    if len(key) < 16:
        raise ValueError("HMAC key must be at least 16 bytes")
    return key
