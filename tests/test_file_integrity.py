"""Final-file integrity: the received file must be bit-identical (SHA-256)."""

import hashlib
import os
import tempfile
import unittest

from protocol.reliability import sha256_file
from tests.harness import transfer
from tests.test_reliability import random_bytes


class TestFileIntegrity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def check(self, data, **kw):
        res, r, _ = transfer(self.tmp, data, **kw)
        self.assertTrue(res.success, res.error)
        path, status = r.completed[-1]
        self.assertEqual(status, "OK")
        self.assertEqual(sha256_file(path), hashlib.sha256(data).digest())
        self.assertFalse([n for n in os.listdir(os.path.dirname(path)) if n.endswith(".part")])
        return res

    def test_empty_file(self):
        res = self.check(b"")
        self.assertEqual(res.data_packets_sent, 0)

    def test_one_byte(self):
        self.check(b"\x00")

    def test_exact_multiple_of_chunk_size(self):
        self.check(random_bytes(1200 * 5, 1))

    def test_all_byte_values(self):
        self.check(bytes(range(256)) * 37)

    def test_large_file(self):
        self.check(random_bytes(8 * 1024 * 1024, 2))

    def test_large_file_through_lossy_network(self):
        self.check(random_bytes(1024 * 1024, 3), profile="lossy", seed=8)

    def test_sha256_helper(self):
        p = os.path.join(self.tmp, "f")
        with open(p, "wb") as f:
            f.write(b"abc")
        self.assertEqual(sha256_file(p).hex(),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
