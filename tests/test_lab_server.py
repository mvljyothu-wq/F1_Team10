"""Live lab server: health, input validation, a real transfer streamed over SSE, static files."""

import json
import threading
import unittest
import urllib.error
import urllib.request

import lab_server


class TestLabServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stop = threading.Event()
        got = threading.Event()
        cls.port = None

        def ready(p):
            cls.port = p
            got.set()
        cls.th = threading.Thread(target=lab_server.serve, args=(0, cls.stop, ready, (9830, 9831)), daemon=True)
        cls.th.start()
        assert got.wait(10), "server did not start"
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.stop.set()
        cls.th.join(timeout=10)

    def post(self, body):
        req = urllib.request.Request(self.base + "/api/run", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health(self):
        with urllib.request.urlopen(self.base + "/api/health", timeout=5) as r:
            d = json.loads(r.read())
        self.assertTrue(d["ok"])
        self.assertIn("lossy", d["profiles"])
        self.assertIn("rto", d["modes"])

    def test_rejects_bad_input(self):
        for body in ({"profile": "../etc"}, {"mode": "evil"}, {"size_kb": 999999}, {"seed": -1}):
            code, d = self.post(body)
            self.assertEqual(code, 400, body)

    def test_static_files_cannot_escape_site_dir(self):
        for path in ("/../../lab_server.py", "/%2e%2e/%2e%2e/README.md", "/nope.html"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(self.base + path, timeout=5)
            self.assertEqual(ctx.exception.code, 404)

    def test_live_transfer_streams_events(self):
        code, d = self.post({"profile": "lossy", "mode": "rto", "size_kb": 256, "seed": 3, "rack": True})
        self.assertEqual(code, 200, d)
        kinds = []
        with urllib.request.urlopen(f"{self.base}/api/events?run={d['run']}", timeout=60) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("event:"):
                    kinds.append(line.split(":", 1)[1].strip())
                if line.startswith("data:") and kinds and kinds[-1] == "done":
                    done = json.loads(line[5:])
        self.assertEqual(kinds[0], "start")
        self.assertIn("sample", kinds)
        self.assertIn("progress", kinds)
        self.assertEqual(kinds[-1], "done")
        self.assertTrue(done["result"]["success"])
        self.assertTrue(done["verified"])


if __name__ == "__main__":
    unittest.main()
