"""Packet format for the reliable UDP file-transfer protocol.

Wire layout (big-endian, 24-byte header, see docs/DESIGN.md section 5):

    magic(2) version(1) msg_type(1) flags(1) reserved(1) payload_len(2)
    transfer_id(4) seq(4) ack(4) checksum(4) | payload | [HMAC tag (16)]

checksum = CRC-32 over: header (with the checksum field set to 0) + payload
           + tag (if present). Detects accidental corruption only.

Optional authentication: when both ends share a key, flags has FLAG_HMAC set
and a 16-byte truncated HMAC-SHA256 follows the payload. The tag covers the
header (checksum field zeroed) and the payload. payload_len never includes
the tag.

This module has no socket code, so it can be tested in isolation.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

MAGIC = 0x5255          # ASCII "RU"
VERSION = 1

HEADER_FORMAT = "!HBBBBHIIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)   # 24 bytes
CHECKSUM_OFFSET = 20                           # byte offset of the checksum field

FLAG_HMAC = 0x01
TAG_SIZE = 16                                  # truncated HMAC-SHA256

MAX_PAYLOAD = 1200      # data bytes per packet; see DESIGN.md section 5
MAX_DATAGRAM = HEADER_SIZE + MAX_PAYLOAD + TAG_SIZE   # 1240; +28 (UDP+IPv4) < 1280

U32_MAX = 0xFFFFFFFF


class MsgType(IntEnum):
    SYN = 1
    SYN_ACK = 2
    DATA = 3
    ACK = 4
    FIN = 5
    FIN_ACK = 6
    ERROR = 7


class Status(IntEnum):
    """Status byte carried in FIN_ACK and ERROR payloads."""
    OK = 0
    HASH_MISMATCH = 1
    INCOMPLETE = 2
    REJECTED = 3
    INTERNAL_ERROR = 4


class PacketError(ValueError):
    """Raised for any malformed, corrupted, unauthenticated or out-of-range packet."""


@dataclass(frozen=True)
class Packet:
    msg_type: MsgType
    transfer_id: int
    seq: int = 0
    ack: int = 0
    payload: bytes = b""


def _crc32(*parts: bytes) -> int:
    crc = 0
    for part in parts:
        crc = zlib.crc32(part, crc)
    return crc & U32_MAX


def _tag(key: bytes, header_zeroed: bytes, payload: bytes) -> bytes:
    return hmac.new(key, header_zeroed + payload, hashlib.sha256).digest()[:TAG_SIZE]


def _check_u32(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= U32_MAX:
        raise PacketError(f"{name} must be an integer in [0, 2^32-1], got {value!r}")


def encode(pkt: Packet, key: Optional[bytes] = None, max_payload: int = MAX_PAYLOAD) -> bytes:
    """Serialize a Packet to bytes, validating every field first.

    If key is given, the packet is authenticated with a truncated HMAC-SHA256.
    """
    try:
        msg_type = MsgType(pkt.msg_type)
    except ValueError:
        raise PacketError(f"unknown msg_type {pkt.msg_type!r}") from None
    _check_u32("transfer_id", pkt.transfer_id)
    _check_u32("seq", pkt.seq)
    _check_u32("ack", pkt.ack)
    if not isinstance(pkt.payload, (bytes, bytearray)):
        raise PacketError("payload must be bytes")
    if len(pkt.payload) > max_payload:
        raise PacketError(f"payload too large: {len(pkt.payload)} > {max_payload}")

    payload = bytes(pkt.payload)
    flags = FLAG_HMAC if key else 0
    header_zeroed = struct.pack(HEADER_FORMAT, MAGIC, VERSION, msg_type, flags, 0,
                                len(payload), pkt.transfer_id, pkt.seq, pkt.ack, 0)
    tag = _tag(key, header_zeroed, payload) if key else b""
    checksum = _crc32(header_zeroed, payload, tag)
    header = header_zeroed[:CHECKSUM_OFFSET] + struct.pack("!I", checksum)
    return header + payload + tag


def decode(data: bytes, key: Optional[bytes] = None, max_payload: int = MAX_PAYLOAD) -> Packet:
    """Parse and validate bytes received from the network.

    Raises PacketError for anything that is not a well-formed packet.
    Never raises any other exception type for bad input.
    If key is given, packets without a valid HMAC tag are rejected.
    If key is None, authenticated packets are rejected (no way to verify them).
    """
    if len(data) < HEADER_SIZE:
        raise PacketError(f"datagram too short: {len(data)} bytes")
    if len(data) > HEADER_SIZE + max_payload + TAG_SIZE:
        raise PacketError(f"datagram too long: {len(data)} bytes")

    (magic, version, msg_type, flags, reserved, payload_len,
     transfer_id, seq, ack, checksum) = struct.unpack_from(HEADER_FORMAT, data)

    if magic != MAGIC:
        raise PacketError(f"bad magic 0x{magic:04x}")
    if version != VERSION:
        raise PacketError(f"unsupported version {version}")
    if flags not in (0, FLAG_HMAC) or reserved != 0:
        raise PacketError("invalid flags/reserved field")
    if payload_len > max_payload:
        raise PacketError(f"payload_len {payload_len} exceeds limit {max_payload}")
    tag_len = TAG_SIZE if flags & FLAG_HMAC else 0
    if HEADER_SIZE + payload_len + tag_len != len(data):
        raise PacketError(f"length mismatch: payload_len {payload_len}, "
                          f"datagram {len(data)} bytes")

    header_zeroed = bytes(data[:CHECKSUM_OFFSET]) + b"\x00\x00\x00\x00"
    payload = bytes(data[HEADER_SIZE:HEADER_SIZE + payload_len])
    tag = bytes(data[HEADER_SIZE + payload_len:])
    if _crc32(header_zeroed, payload, tag) != checksum:
        raise PacketError("checksum mismatch")

    # Authentication is checked after the CRC: corruption is reported as
    # corruption, and only intact packets reach the (slower) HMAC check.
    if key:
        if not tag_len:
            raise PacketError("unauthenticated packet rejected (key configured)")
        if not hmac.compare_digest(tag, _tag(key, header_zeroed, payload)):
            raise PacketError("HMAC verification failed")
    elif tag_len:
        raise PacketError("authenticated packet received but no key configured")

    try:
        msg_type = MsgType(msg_type)
    except ValueError:
        raise PacketError(f"unknown msg_type {msg_type}") from None

    return Packet(msg_type, transfer_id, seq, ack, payload)


def peek(data: bytes) -> Optional[Tuple[int, int]]:
    """Return (msg_type, seq) without validating anything, or None.

    Used only by the impairment proxy, which plays the role of the network
    and must not depend on keys or checksums.
    """
    if len(data) < HEADER_SIZE:
        return None
    if struct.unpack_from("!H", data, 0)[0] != MAGIC:
        return None
    return data[3], struct.unpack_from("!I", data, 12)[0]


# ---------------------------------------------------------------------------
# SYN payload: file metadata + sender window
# ---------------------------------------------------------------------------

SYN_FIXED_FORMAT = "!QIHH32sB"   # file_size, total_chunks, chunk_size, window, sha256, name_len
SYN_FIXED_SIZE = struct.calcsize(SYN_FIXED_FORMAT)   # 49 bytes
MAX_FILENAME_BYTES = 255
MAX_WINDOW = 4096


@dataclass(frozen=True)
class SynInfo:
    file_size: int
    total_chunks: int
    chunk_size: int
    window: int
    sha256: bytes
    filename: str


def expected_chunks(file_size: int, chunk_size: int) -> int:
    """Number of chunks for a file; an empty file has 0 chunks."""
    return (file_size + chunk_size - 1) // chunk_size


def chunk_length(info: SynInfo, seq: int) -> int:
    """Exact payload length that DATA packet `seq` must have."""
    if seq < info.total_chunks - 1:
        return info.chunk_size
    return info.file_size - (info.total_chunks - 1) * info.chunk_size


def _validate_syn_numbers(file_size: int, total_chunks: int, chunk_size: int, window: int) -> None:
    if not 1 <= chunk_size <= MAX_PAYLOAD:
        raise PacketError(f"chunk_size must be in 1..{MAX_PAYLOAD}")
    if not 1 <= window <= MAX_WINDOW:
        raise PacketError(f"window must be in 1..{MAX_WINDOW}")
    if file_size < 0:
        raise PacketError("file_size must be non-negative")
    if total_chunks != expected_chunks(file_size, chunk_size):
        raise PacketError("total_chunks inconsistent with file_size/chunk_size")
    _check_u32("total_chunks", total_chunks)


def encode_syn(info: SynInfo) -> bytes:
    name = info.filename.encode("utf-8")
    if not 1 <= len(name) <= MAX_FILENAME_BYTES:
        raise PacketError("filename must be 1..255 bytes of UTF-8")
    if len(info.sha256) != 32:
        raise PacketError("sha256 digest must be 32 bytes")
    _validate_syn_numbers(info.file_size, info.total_chunks, info.chunk_size, info.window)
    return struct.pack(SYN_FIXED_FORMAT, info.file_size, info.total_chunks,
                       info.chunk_size, info.window, info.sha256, len(name)) + name


def decode_syn(payload: bytes) -> SynInfo:
    """Parse a SYN payload. Does not sanitize the filename for filesystem use;
    the receiver must do that before creating any file."""
    if len(payload) < SYN_FIXED_SIZE:
        raise PacketError("SYN payload too short")
    file_size, total_chunks, chunk_size, window, sha256, name_len = \
        struct.unpack_from(SYN_FIXED_FORMAT, payload)
    name_bytes = payload[SYN_FIXED_SIZE:]
    if name_len == 0 or len(name_bytes) != name_len:
        raise PacketError("SYN filename length mismatch")
    _validate_syn_numbers(file_size, total_chunks, chunk_size, window)
    try:
        filename = name_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise PacketError("SYN filename is not valid UTF-8") from None
    return SynInfo(file_size, total_chunks, chunk_size, window, bytes(sha256), filename)


# ---------------------------------------------------------------------------
# FIN_ACK / ERROR payload: status byte + optional short message
# ---------------------------------------------------------------------------

MAX_STATUS_MESSAGE = 200


def encode_status(status: Status, message: str = "") -> bytes:
    msg = message.encode("utf-8")[:MAX_STATUS_MESSAGE]
    return bytes([Status(status)]) + msg


def decode_status(payload: bytes) -> Tuple[Status, str]:
    """Return (Status, message)."""
    if len(payload) < 1:
        raise PacketError("status payload empty")
    if len(payload) > 1 + MAX_STATUS_MESSAGE:
        raise PacketError("status message too long")
    try:
        status = Status(payload[0])
    except ValueError:
        raise PacketError(f"unknown status {payload[0]}") from None
    return status, payload[1:].decode("utf-8", errors="replace")
