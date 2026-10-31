"""End-to-end transfers over loopback, with and without network impairment."""

import os
import random
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

import protocol.reliability as rel
from protocol.packet import (
    MsgType, Packet, Status, SynInfo, decode, decode_status, encode, encode_syn,
    expected_chunks,
)
from protocol.reliability import Receiver, ReceiverConfig, SenderConfig, safe_filename, send_file
from tests.harness import running_proxy, running_receiver, transfer, write_file


def random_bytes(n, seed=0):
    return random.Random(seed).randbytes(n) if hasattr(random.Random, "randbytes") \
        else bytes(random.Random(seed).getrandbits(8) for _ in range(n))


def read(path):
    with open(path, "rb") as f:
        return f.read()


class TransferTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def assertDelivered(self, res, receiver, data):
        self.assertTrue(res.success, res.error)
        self.assertEqual(res.status, "OK")
        path, status = receiver.completed[-1]
        self.assertEqual(status, "OK")
        self.assertEqual(read(path), data)


class TestNormalOperation(TransferTest):
    def test_binary_file_no_impairment(self):
        data = random_bytes(500_000)
        res, r, _ = transfer(self.tmp, data)
        self.assertDelivered(res, r, data)
        self.assertEqual(res.retransmissions, 0)
        self.assertGreater(res.rtt_samples, 100)

    def test_stop_and_wait_window_of_one(self):
        data = random_bytes(30_000, 1)
        res, r, _ = transfer(self.tmp, data, window=1)
        self.assertDelivered(res, r, data)

    def test_small_payload_size(self):
        data = random_bytes(20_011, 2)
        res, r, _ = transfer(self.tmp, data, payload_size=100)
        self.assertDelivered(res, r, data)
        self.assertEqual(res.data_packets_sent, expected_chunks(len(data), 100))

    def test_duplicate_filename_gets_suffix(self):
        src = write_file(os.path.join(self.tmp, "same.txt"), b"hello")
        with running_receiver(os.path.join(self.tmp, "out")) as r:
            for _ in range(2):
                self.assertTrue(send_file(src, SenderConfig(port=r.port)).success)
        names = sorted(os.path.basename(p) for p, _ in r.completed)
        self.assertEqual(names, ["same (1).txt", "same.txt"])

    def test_name_reusable_after_file_removed(self):
        # Regression: a finished transfer in its linger period must not keep the
        # name reserved once its file is gone (found while running experiments).
        src = write_file(os.path.join(self.tmp, "again.txt"), b"hello")
        with running_receiver(os.path.join(self.tmp, "out")) as r:
            self.assertTrue(send_file(src, SenderConfig(port=r.port)).success)
            os.remove(r.completed[-1][0])
            self.assertTrue(send_file(src, SenderConfig(port=r.port)).success)
        self.assertEqual(os.path.basename(r.completed[-1][0]), "again.txt")


class TestAdverseNetwork(TransferTest):
    def test_random_loss(self):
        data = random_bytes(300_000, 3)
        res, r, stats = transfer(self.tmp, data, profile="lossy", seed=4)
        self.assertDelivered(res, r, data)
        self.assertGreater(stats["data_copies_lost"], 0)
        self.assertGreater(res.retransmissions, 0)
        self.assertGreater(res.timeouts, 0)

    def test_burst_loss(self):
        data = random_bytes(300_000, 4)
        res, r, stats = transfer(self.tmp, data, profile="bursty_loss", seed=2)
        self.assertDelivered(res, r, data)

    def test_duplicates_reordering_corruption_delayed_acks(self):
        data = random_bytes(400_000, 5)
        res, r, stats = transfer(self.tmp, data, profile="dup_reorder", seed=6)
        self.assertDelivered(res, r, data)
        self.assertGreater(stats["counts"]["fwd"]["duplicated"], 0)
        self.assertGreater(stats["counts"]["fwd"]["reordered"], 0)
        corrupted = stats["counts"]["fwd"]["corrupted"] + stats["counts"]["rev"]["corrupted"]
        self.assertGreater(corrupted, 0)
        # corrupted datagrams were caught by the CRC and dropped (the file is still
        # correct). Exact counts can differ: a duplicated corrupt packet is seen
        # twice, and late copies may arrive after the sender has closed.
        self.assertGreater(r.stats["invalid"] + res.invalid_received, 0)

    def test_rack_repairs_losses_before_timers(self):
        data = random_bytes(400_000, 11)
        res, r, stats = transfer(self.tmp, data, profile="lossy", seed=4, fast_retransmit=True)
        self.assertDelivered(res, r, data)
        self.assertGreater(res.fast_retransmits, 0)
        self.assertLessEqual(res.fast_retransmits, res.retransmissions)

    def test_rack_under_reordering_still_correct(self):
        data = random_bytes(300_000, 12)
        res, r, _ = transfer(self.tmp, data, profile="dup_reorder", seed=3, fast_retransmit=True)
        self.assertDelivered(res, r, data)

    def test_event_hook_reports_samples_and_survives_errors(self):
        seen = []
        def hook(kind, d):
            seen.append(kind)
            if len(seen) == 5:
                raise RuntimeError("observer bug")            # must not break the transfer
        data = random_bytes(200_000, 13)
        src = write_file(os.path.join(self.tmp, "h.bin"), data)
        with running_receiver(os.path.join(self.tmp, "out")) as r:
            res = send_file(src, SenderConfig(port=r.port), events=hook)
        self.assertTrue(res.success)
        self.assertIn("sample", seen)

    def test_heavy_jitter(self):
        data = random_bytes(200_000, 6)
        res, r, _ = transfer(self.tmp, data, profile="jitter_heavy", seed=1)
        self.assertDelivered(res, r, data)


class TestFailures(TransferTest):
    def test_no_receiver_fails_quickly_and_never_reports_success(self):
        src = write_file(os.path.join(self.tmp, "x.bin"), b"data")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("127.0.0.1", 0))                 # a port that nobody answers on
        port = s.getsockname()[1]
        t0 = time.monotonic()
        res = send_file(src, SenderConfig(port=port, rto_initial=0.05, handshake_retries=2))
        s.close()
        self.assertFalse(res.success)
        self.assertIn("no SYN_ACK", res.error)
        self.assertLess(time.monotonic() - t0, 2.0)

    def test_receiver_interrupted_mid_transfer(self):
        src = write_file(os.path.join(self.tmp, "big.bin"), random_bytes(3_000_000, 7))
        out = os.path.join(self.tmp, "out")
        r = Receiver(ReceiverConfig(host="127.0.0.1", port=0, output_dir=out))
        stop = threading.Event()
        th = threading.Thread(target=r.serve, args=(stop,))
        th.start()
        with running_proxy(r.port, "high_latency") as p:
            threading.Timer(0.8, stop.set).start()       # receiver "crashes" after 0.8 s
            res = send_file(src, SenderConfig(port=p.port, max_retries=3, rto_max=0.5))
        th.join()
        self.assertFalse(res.success)
        self.assertIn("not acknowledged", res.error)
        self.assertEqual(os.listdir(out), [])            # partial file removed

    def test_receiver_discards_idle_transfer(self):
        out = os.path.join(self.tmp, "out")
        with running_receiver(out, idle_timeout=0.3) as r:
            info = SynInfo(2400, 2, 1200, 4, b"\x00" * 32, "half.bin")
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.sendto(encode(Packet(MsgType.SYN, 77, payload=encode_syn(info))), ("127.0.0.1", r.port))
            self.assertEqual(decode(s.recv(2048)).msg_type, MsgType.SYN_ACK)
            s.sendto(encode(Packet(MsgType.DATA, 77, seq=0, payload=b"a" * 1200)), ("127.0.0.1", r.port))
            self.assertEqual(decode(s.recv(2048)).ack, 1)
            self.assertEqual(len(os.listdir(out)), 1)    # the .part file exists
            time.sleep(1.0)                              # sender "crashed"
            s.close()
            self.assertEqual(os.listdir(out), [])
            self.assertIn((None, "ABORTED_IDLE"), r.completed)

    def test_hash_mismatch_is_detected_and_file_discarded(self):
        data = random_bytes(10_000, 8)
        with mock.patch.object(rel, "sha256_file", return_value=b"\x11" * 32):
            res, r, _ = transfer(self.tmp, data)
        self.assertFalse(res.success)
        self.assertEqual(res.status, "HASH_MISMATCH")
        self.assertEqual(os.listdir(os.path.join(self.tmp, "out")), [])

    def test_missing_ml_model_falls_back_and_still_works(self):
        data = random_bytes(50_000, 9)
        res, r, _ = transfer(self.tmp, data, timeout_mode="ml", model_path="/nonexistent.json")
        self.assertDelivered(res, r, data)
        self.assertIsNotNone(res.model_error)
        self.assertEqual(res.ml_decisions, 0)
        self.assertGreater(res.ml_fallbacks.get("no_model", 0), 0)


class TestReceiverValidation(TransferTest):
    def raw_syn(self, port, info, tid=5, sock=None):
        s = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(1)
            s.sendto(encode(Packet(MsgType.SYN, tid, payload=encode_syn(info))), ("127.0.0.1", port))
            return decode(s.recv(2048))
        finally:
            if sock is None:
                s.close()

    def test_path_traversal_rejected(self):
        out = os.path.join(self.tmp, "out")
        with running_receiver(out) as r:
            for name in ("../evil.txt", "a/b.txt", "..", "x\x00y", "tab\tname"):
                info = SynInfo(10, 1, 1200, 4, b"\x00" * 32, name)
                reply = self.raw_syn(r.port, info)
                self.assertEqual(reply.msg_type, MsgType.ERROR, name)
                self.assertEqual(decode_status(reply.payload)[0], Status.REJECTED)
        self.assertEqual(os.listdir(out), [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "evil.txt")))

    def test_size_and_window_limits(self):
        with running_receiver(os.path.join(self.tmp, "out"), max_file_size=1000, max_window=16) as r:
            big = SynInfo(5000, expected_chunks(5000, 1200), 1200, 4, b"\x00" * 32, "a")
            self.assertEqual(self.raw_syn(r.port, big).msg_type, MsgType.ERROR)
            wide = SynInfo(10, 1, 1200, 64, b"\x00" * 32, "b")
            self.assertEqual(self.raw_syn(r.port, wide, tid=6).msg_type, MsgType.ERROR)

    def test_transfer_id_reuse_with_different_metadata_rejected(self):
        with running_receiver(os.path.join(self.tmp, "out")) as r:
            a = SynInfo(10, 1, 1200, 4, b"\x00" * 32, "a.bin")
            b = SynInfo(20, 1, 1200, 4, b"\x00" * 32, "b.bin")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                self.assertEqual(self.raw_syn(r.port, a, 9, s).msg_type, MsgType.SYN_ACK)
                self.assertEqual(self.raw_syn(r.port, a, 9, s).msg_type, MsgType.SYN_ACK)  # duplicate
                self.assertEqual(self.raw_syn(r.port, b, 9, s).msg_type, MsgType.ERROR)
            # The same transfer id from a different address is silently ignored:
            # a transfer is bound to the address that started it.
            with self.assertRaises(socket.timeout):
                self.raw_syn(r.port, a, 9)
            self.assertEqual(r.stats["wrong_source"], 1)

    def test_safe_filename(self):
        self.assertEqual(safe_filename("report.pdf"), "report.pdf")
        self.assertEqual(safe_filename("résumé.txt"), "résumé.txt")
        for bad in ("", ".", "..", "../x", "a/b", "a\\b", "x\x01", "a" * 256):
            with self.assertRaises(ValueError, msg=repr(bad)):
                safe_filename(bad)


class TestAuthentication(TransferTest):
    def test_hmac_transfer(self):
        key = b"shared-secret-key-for-tests-0123"
        data = random_bytes(100_000, 10)
        res, r, _ = transfer(self.tmp, data, profile="lossy", key=key,
                             receiver_kwargs={"key": key})
        self.assertDelivered(res, r, data)

    def test_key_mismatch_fails(self):
        res, r, _ = transfer(self.tmp, b"abc", key=b"key-one", rto_initial=0.05,
                             handshake_retries=2, receiver_kwargs={"key": b"key-two"})
        self.assertFalse(res.success)
        self.assertGreater(r.stats["invalid"], 0)


if __name__ == "__main__":
    unittest.main()
