"""Tests for protocol/packet.py. Run: python -m unittest -v (or pytest)."""

import hashlib
import random
import struct
import unittest
import zlib

from protocol.packet import (
    HEADER_SIZE, MAX_DATAGRAM, MAX_PAYLOAD, TAG_SIZE, MsgType, Packet, PacketError,
    Status, SynInfo, chunk_length, decode, decode_status, decode_syn, encode,
    encode_status, encode_syn, expected_chunks, peek,
)

KEY = b"k" * 32


class TestRoundTrip(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(HEADER_SIZE, 24)
        self.assertEqual(MAX_DATAGRAM, 1240)
        self.assertLess(MAX_DATAGRAM + 8 + 20, 1280)   # no IPv6-minimum fragmentation

    def test_every_type_round_trips(self):
        for t in MsgType:
            pkt = Packet(t, transfer_id=0xDEADBEEF, seq=7, ack=3, payload=b"abc")
            self.assertEqual(decode(encode(pkt)), pkt)

    def test_empty_and_max_payload(self):
        for payload in (b"", bytes(range(256)) * 4 + b"x" * (MAX_PAYLOAD - 1024)):
            pkt = Packet(MsgType.DATA, 1, 0, 0, payload)
            self.assertEqual(decode(encode(pkt)).payload, payload)

    def test_extreme_field_values(self):
        pkt = Packet(MsgType.ACK, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(decode(encode(pkt)), pkt)

    def test_byte_order_is_big_endian(self):
        raw = encode(Packet(MsgType.DATA, transfer_id=1, seq=0x01020304))
        self.assertEqual(raw[0:2], b"RU")
        self.assertEqual(raw[12:16], b"\x01\x02\x03\x04")

    def test_peek(self):
        raw = encode(Packet(MsgType.ACK, 5, seq=77))
        self.assertEqual(peek(raw), (MsgType.ACK, 77))
        self.assertIsNone(peek(b"short"))
        self.assertIsNone(peek(b"\x00" * 30))


class TestEncodeValidation(unittest.TestCase):
    def test_oversize_payload(self):
        with self.assertRaises(PacketError):
            encode(Packet(MsgType.DATA, 1, payload=b"x" * (MAX_PAYLOAD + 1)))

    def test_out_of_range_fields(self):
        for kwargs in ({"seq": -1}, {"seq": 2**32}, {"ack": 2**32}):
            with self.assertRaises(PacketError):
                encode(Packet(MsgType.DATA, 1, **kwargs))
        with self.assertRaises(PacketError):
            encode(Packet(MsgType.DATA, 2**32))

    def test_unknown_type(self):
        with self.assertRaises(PacketError):
            encode(Packet(99, 1))


class TestDecodeRejects(unittest.TestCase):
    def setUp(self):
        self.good = encode(Packet(MsgType.DATA, 42, seq=5, payload=b"hello world"))

    def test_truncated(self):
        for n in (0, 1, HEADER_SIZE - 1):
            with self.assertRaises(PacketError):
                decode(self.good[:n])
        with self.assertRaises(PacketError):      # payload cut short
            decode(self.good[:-1])

    def test_too_long(self):
        with self.assertRaises(PacketError):
            decode(b"\x00" * (MAX_DATAGRAM + 1))

    def test_bad_magic_version_flags_reserved(self):
        for offset in (0, 2, 4, 5):
            bad = bytearray(self.good)
            bad[offset] ^= 0x02
            with self.assertRaises(PacketError):
                decode(bytes(bad))

    def test_every_single_bit_flip_is_detected(self):
        # CRC-32 detects all single-bit errors; check it over the whole packet.
        for byte in range(len(self.good)):
            for bit in range(8):
                bad = bytearray(self.good)
                bad[byte] ^= 1 << bit
                with self.assertRaises(PacketError, msg=f"byte {byte} bit {bit}"):
                    decode(bytes(bad))

    def test_unknown_type_with_valid_crc(self):
        header = struct.pack("!HBBBBHIIII", 0x5255, 1, 99, 0, 0, 0, 1, 0, 0, 0)
        crc = zlib.crc32(header) & 0xFFFFFFFF
        with self.assertRaises(PacketError):
            decode(header[:20] + struct.pack("!I", crc))

    def test_random_garbage_never_crashes(self):
        rng = random.Random(1234)
        for _ in range(5000):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 1300)))
            for key in (None, KEY):
                try:
                    decode(data, key)
                except PacketError:
                    pass   # the only acceptable exception


class TestHmac(unittest.TestCase):
    def test_round_trip_with_key(self):
        pkt = Packet(MsgType.DATA, 9, seq=3, payload=b"secret data")
        raw = encode(pkt, KEY)
        self.assertEqual(len(raw), HEADER_SIZE + len(pkt.payload) + TAG_SIZE)
        self.assertEqual(decode(raw, KEY), pkt)

    def test_wrong_key_rejected(self):
        raw = encode(Packet(MsgType.ACK, 9, seq=3), KEY)
        with self.assertRaises(PacketError):
            decode(raw, b"other key")

    def test_key_mismatch_in_either_direction(self):
        with self.assertRaises(PacketError):
            decode(encode(Packet(MsgType.ACK, 1)), KEY)       # unauthenticated, key expected
        with self.assertRaises(PacketError):
            decode(encode(Packet(MsgType.ACK, 1), KEY))       # authenticated, no key

    def test_forged_packet_with_recomputed_crc_rejected(self):
        # An attacker can always fix the CRC; only the HMAC stops forgery.
        raw = bytearray(encode(Packet(MsgType.DATA, 9, seq=3, payload=b"aaaa"), KEY))
        raw[HEADER_SIZE] = ord("b")                           # tamper payload
        raw[20:24] = b"\x00" * 4
        crc = zlib.crc32(bytes(raw)) & 0xFFFFFFFF
        raw[20:24] = struct.pack("!I", crc)
        with self.assertRaises(PacketError) as ctx:
            decode(bytes(raw), KEY)
        self.assertIn("HMAC", str(ctx.exception))

    def test_max_payload_with_tag_fits(self):
        raw = encode(Packet(MsgType.DATA, 1, payload=b"x" * MAX_PAYLOAD), KEY)
        self.assertEqual(len(raw), MAX_DATAGRAM)
        decode(raw, KEY)


class TestSynPayload(unittest.TestCase):
    def make(self, size=2500, chunk=1200, name="report.pdf", window=64):
        return SynInfo(size, expected_chunks(size, chunk), chunk, window,
                       hashlib.sha256(b"x").digest(), name)

    def test_round_trip(self):
        info = self.make()
        self.assertEqual(info.total_chunks, 3)
        self.assertEqual(decode_syn(encode_syn(info)), info)

    def test_chunk_lengths(self):
        info = self.make(size=2500)
        self.assertEqual([chunk_length(info, s) for s in range(3)], [1200, 1200, 100])
        exact = self.make(size=2400)
        self.assertEqual([chunk_length(exact, s) for s in range(2)], [1200, 1200])

    def test_empty_file_has_zero_chunks(self):
        info = self.make(size=0)
        self.assertEqual(info.total_chunks, 0)
        self.assertEqual(decode_syn(encode_syn(info)), info)

    def test_unicode_filename(self):
        info = self.make(name="résumé_ファイル.txt")
        self.assertEqual(decode_syn(encode_syn(info)).filename, info.filename)

    def test_rejects_inconsistent_metadata(self):
        raw = bytearray(encode_syn(self.make()))
        raw[8:12] = struct.pack("!I", 99)          # wrong total_chunks
        with self.assertRaises(PacketError):
            decode_syn(bytes(raw))

    def test_rejects_bad_window(self):
        raw = bytearray(encode_syn(self.make()))
        raw[14:16] = struct.pack("!H", 0)
        with self.assertRaises(PacketError):
            decode_syn(bytes(raw))

    def test_rejects_bad_filename_length(self):
        raw = encode_syn(self.make())
        with self.assertRaises(PacketError):
            decode_syn(raw[:-1])
        with self.assertRaises(PacketError):
            decode_syn(raw + b"x")

    def test_rejects_invalid_utf8(self):
        raw = bytearray(encode_syn(self.make(name="ab")))
        raw[-2:] = b"\xff\xfe"
        with self.assertRaises(PacketError):
            decode_syn(bytes(raw))

    def test_encode_rejects_bad_input(self):
        for kwargs in ({"name": ""}, {"name": "a" * 256}, {"chunk": MAX_PAYLOAD + 1},
                       {"window": 0}, {"window": 5000}):
            with self.assertRaises(PacketError, msg=str(kwargs)):
                encode_syn(self.make(**kwargs))

    def test_syn_fits_in_one_packet(self):
        info = self.make(name="a" * 255)
        pkt = Packet(MsgType.SYN, 1, payload=encode_syn(info))
        self.assertLessEqual(len(encode(pkt, KEY)), MAX_DATAGRAM)


class TestStatusPayload(unittest.TestCase):
    def test_round_trip(self):
        self.assertEqual(decode_status(encode_status(Status.OK)), (Status.OK, ""))
        self.assertEqual(decode_status(encode_status(Status.HASH_MISMATCH, "bad")),
                         (Status.HASH_MISMATCH, "bad"))

    def test_rejects(self):
        with self.assertRaises(PacketError):
            decode_status(b"")
        with self.assertRaises(PacketError):
            decode_status(b"\x63")


if __name__ == "__main__":
    unittest.main()
