"""Timeout estimation: RFC 6298 estimator, backoff, features, ML fallback and safety."""

import json
import math
import os
import tempfile
import unittest

from timeout.conventional import ConventionalRTO
from timeout.features import FEATURE_NAMES, History, compute_features, percentile
from timeout.manager import TimeoutManager
from timeout.ml_estimator import (
    ForestModel, MLTimeoutEstimator, ModelError, apply_safety, load_estimator,
)


def tiny_model_doc(target="correction", leaf_values=(0.0, math.log(2.0)), margin=1.0):
    """One tree: if rttvar_rel <= 0.1 predict leaf_values[0] else leaf_values[1]."""
    f = FEATURE_NAMES.index("rttvar_rel")
    return {
        "format": "rto-forest", "format_version": 1, "feature_names": list(FEATURE_NAMES),
        "target": target, "margin": margin,
        "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [f, -2, -2],
                   "threshold": [0.1, -2.0, -2.0], "value": [0.0, leaf_values[0], leaf_values[1]]}],
    }


def features(rttvar_rel=0.05):
    x = [0.0] * len(FEATURE_NAMES)
    x[FEATURE_NAMES.index("rttvar_rel")] = rttvar_rel
    return x


class TestConventionalRTO(unittest.TestCase):
    def test_hand_computed_values(self):
        e = ConventionalRTO(rto_min=0.001, rto_max=60)
        self.assertAlmostEqual(e.update(0.1), 0.1 + 4 * 0.05)          # 0.3
        self.assertAlmostEqual(e.srtt, 0.1)
        self.assertAlmostEqual(e.rttvar, 0.05)
        # RTTVAR uses the OLD SRTT: 0.75*0.05 + 0.25*|0.1-0.2| = 0.0625
        e.update(0.2)
        self.assertAlmostEqual(e.rttvar, 0.0625)
        self.assertAlmostEqual(e.srtt, 0.1125)
        self.assertAlmostEqual(e.rto, 0.1125 + 0.25)

    def test_initial_and_clamping(self):
        e = ConventionalRTO(rto_min=0.03, rto_max=2.0, rto_initial=1.0)
        self.assertEqual(e.rto, 1.0)                 # before any sample
        e.update(0.001)
        self.assertEqual(e.rto, 0.03)                # clamped to rto_min
        e2 = ConventionalRTO(rto_min=0.03, rto_max=2.0)
        e2.update(5.0)
        self.assertEqual(e2.rto, 2.0)                # clamped to rto_max

    def test_granularity_floor(self):
        e = ConventionalRTO(rto_min=0.0001, rto_max=10, granularity=0.001)
        for _ in range(200):
            e.update(0.01)                           # RTTVAR decays towards 0
        self.assertAlmostEqual(e.rto, e.srtt + 0.001, places=6)

    def test_converges_to_constant_rtt(self):
        e = ConventionalRTO(rto_min=0.001)
        for _ in range(100):
            e.update(0.05)
        self.assertAlmostEqual(e.srtt, 0.05, places=4)

    def test_rejects_invalid_samples(self):
        e = ConventionalRTO()
        for bad in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                e.update(bad)

    def test_bad_bounds(self):
        with self.assertRaises(ValueError):
            ConventionalRTO(rto_min=2.0, rto_max=1.0)


class TestBackoff(unittest.TestCase):
    def setUp(self):
        self.tm = TimeoutManager("rto", rto_min=0.01, rto_max=4.0, rto_initial=0.1)

    def test_doubles_per_round(self):
        epoch = self.tm.epoch
        self.tm.on_timeout(1.0, epoch)
        self.assertAlmostEqual(self.tm.current_rto(), 0.2)
        self.tm.on_timeout(1.2, self.tm.epoch)
        self.assertAlmostEqual(self.tm.current_rto(), 0.4)

    def test_burst_of_timeouts_from_same_round_doubles_once(self):
        epoch = self.tm.epoch
        for _ in range(50):                          # 50 packets lost in one outage
            self.tm.on_timeout(1.0, epoch)
        self.assertEqual(self.tm.backoff_level, 1)
        self.assertEqual(self.tm.timeouts, 50)

    def test_stale_packet_cannot_raise_level_after_reset(self):
        old_epoch = self.tm.epoch
        self.tm.on_timeout(1.0, old_epoch)           # level 1
        old_epoch = self.tm.epoch
        self.tm.on_timeout(1.1, old_epoch)           # level 2
        stale = self.tm.epoch                        # packet retransmitted at level 2
        self.assertTrue(self.tm.on_rtt_sample(1.2, 0.05))   # reset to 0
        self.tm.on_timeout(1.3, stale)               # its old timer fires
        self.assertEqual(self.tm.backoff_level, 0)

    def test_capped_at_rto_max(self):
        for _ in range(20):
            self.tm.on_timeout(1.0, self.tm.epoch)
        self.assertEqual(self.tm.current_rto(), 4.0)

    def test_sample_without_backoff_reports_no_reset(self):
        self.assertFalse(self.tm.on_rtt_sample(0.0, 0.05))


class TestFeatures(unittest.TestCase):
    def test_percentile_matches_numpy_definition(self):
        vals = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertAlmostEqual(percentile(vals, 90), 9.1)
        self.assertAlmostEqual(percentile(vals, 50), 5.5)
        self.assertEqual(percentile([3.0], 95), 3.0)

    def test_vector_shape_and_values(self):
        h = History()
        for i in range(20):
            h.add_sample(i * 0.01, 0.05 + 0.001 * i)       # rising RTT
            h.add_transmission(i % 10 == 0)
        x = compute_features(h, srtt=0.06, rttvar=0.006, now=0.2)
        self.assertEqual(len(x), len(FEATURE_NAMES))
        d = dict(zip(FEATURE_NAMES, x))
        self.assertAlmostEqual(d["log_srtt"], math.log(0.06))
        self.assertAlmostEqual(d["rttvar_rel"], 0.1)
        self.assertAlmostEqual(d["slope"], 0.1, places=6)  # 1 ms per 10 ms
        self.assertEqual(d["n_samples"], 16)
        self.assertEqual(d["since_timeout_rel"], 1000.0)   # no timeout yet
        self.assertAlmostEqual(d["retx_ratio"], 2 / 20)
        self.assertTrue(all(math.isfinite(v) for v in x))

    def test_requires_history(self):
        with self.assertRaises(ValueError):
            compute_features(History(), 0.05, 0.01, 0.0)


class TestSafetyPolicy(unittest.TestCase):
    def test_bounds(self):
        kw = dict(rto_conv=0.2, recent_max_rtt=0.05, rto_min=0.03, rto_max=10, band_lo=0.5, band_hi=2.0)
        self.assertAlmostEqual(apply_safety(0.01, **kw), 0.1)    # floored at 0.5 x RTO_conv
        self.assertAlmostEqual(apply_safety(5.0, **kw), 0.4)     # capped at 2 x RTO_conv
        self.assertAlmostEqual(apply_safety(0.25, **kw), 0.25)   # inside band: unchanged

    def test_never_below_recent_max_rtt(self):
        r = apply_safety(0.05, rto_conv=0.1, recent_max_rtt=0.3, rto_min=0.03, rto_max=10,
                         band_lo=0.5, band_hi=2.0)
        self.assertAlmostEqual(r, 0.3)

    def test_absolute_limits(self):
        r = apply_safety(100, rto_conv=8, recent_max_rtt=0, rto_min=0.03, rto_max=10,
                         band_lo=0.5, band_hi=2.0)
        self.assertEqual(r, 10)


class StubModel:
    def __init__(self, value=0.0, target="correction", fail=False):
        self.value, self.target, self.fail, self.margin = value, target, fail, 1.0

    def predict(self, x):
        if self.fail:
            raise RuntimeError("boom")
        return self.value


class TestMLEstimator(unittest.TestCase):
    def est(self, model, **kw):
        return MLTimeoutEstimator(model, rto_min=0.01, rto_max=10, **kw)

    def test_no_model_falls_back(self):
        d = self.est(None).decide(features(), 0.2, 100, 0.05)
        self.assertEqual((d.rto, d.source, d.reason), (0.2, "fallback", "no_model"))

    def test_warmup_falls_back(self):
        d = self.est(StubModel()).decide(features(), 0.2, 3, 0.05)
        self.assertEqual(d.reason, "warmup")

    def test_invalid_and_failing_predictions_fall_back(self):
        self.assertEqual(self.est(StubModel(float("nan"))).decide(features(), 0.2, 100, 0.05).reason,
                         "invalid_prediction")
        self.assertEqual(self.est(StubModel(fail=True)).decide(features(), 0.2, 100, 0.05).reason,
                         "predict_error")

    def test_correction_target(self):
        # log-ratio 0 -> proposal equals RTO_conv
        d = self.est(StubModel(0.0)).decide(features(), 0.2, 100, 0.05)
        self.assertEqual(d.source, "ml")
        self.assertAlmostEqual(d.rto, 0.2)
        # log(0.75) -> 0.15, inside the band
        d = self.est(StubModel(math.log(0.75))).decide(features(), 0.2, 100, 0.05)
        self.assertAlmostEqual(d.rto, 0.15)

    def test_direct_target_and_margin(self):
        d = self.est(StubModel(math.log(0.12), "direct"), margin=1.25).decide(features(), 0.2, 100, 0.05)
        self.assertAlmostEqual(d.proposal, 0.15)
        self.assertAlmostEqual(d.rto, 0.15)

    def test_extreme_prediction_is_bounded(self):
        d = self.est(StubModel(1e6)).decide(features(), 0.2, 100, 0.05)
        self.assertAlmostEqual(d.rto, 0.4)           # 2 x RTO_conv


class TestModelLoading(unittest.TestCase):
    def write(self, doc):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(doc if isinstance(doc, str) else json.dumps(doc))
        self.addCleanup(os.remove, path)
        return path

    def test_valid_tiny_model(self):
        m = ForestModel.load(self.write(tiny_model_doc()))
        self.assertAlmostEqual(m.predict(features(0.05)), 0.0)
        self.assertAlmostEqual(m.predict(features(0.5)), math.log(2.0))

    def test_rejections(self):
        bad_docs = []
        d = tiny_model_doc(); d["feature_names"] = d["feature_names"][:-1]; bad_docs.append(d)
        d = tiny_model_doc(); d["format_version"] = 99; bad_docs.append(d)
        d = tiny_model_doc(); d["target"] = "magic"; bad_docs.append(d)
        d = tiny_model_doc(); d["trees"][0]["left"][0] = 0; bad_docs.append(d)      # cycle
        d = tiny_model_doc(); d["trees"][0]["right"][0] = 7; bad_docs.append(d)     # out of range
        d = tiny_model_doc(); d["trees"][0]["feature"][0] = 99; bad_docs.append(d)
        d = tiny_model_doc(); d["trees"][0]["value"][1] = None; bad_docs.append(d)
        d = tiny_model_doc(); d["trees"] = []; bad_docs.append(d)
        d = tiny_model_doc(); d["margin"] = 100; bad_docs.append(d)
        for doc in bad_docs:
            with self.assertRaises(ModelError):
                ForestModel.load(self.write(doc))
        with self.assertRaises(ModelError):
            ForestModel.load(self.write("{not json"))
        with self.assertRaises(ModelError):
            ForestModel.load("/nonexistent/model.json")

    def test_boosting_model(self):
        doc = tiny_model_doc(leaf_values=(1.0, 3.0))
        doc.update(ensemble="boosting", init=0.5, learning_rate=0.1)
        m = ForestModel.from_dict(doc)
        self.assertAlmostEqual(m.predict(features(0.05)), 0.5 + 0.1 * 1.0)
        self.assertAlmostEqual(m.predict(features(0.5)), 0.5 + 0.1 * 3.0)
        for bad in ({"ensemble": "magic"}, {"ensemble": "boosting", "init": 0.1, "learning_rate": 0},
                    {"ensemble": "boosting", "init": None, "learning_rate": 0.1}):
            d = tiny_model_doc(); d.update(bad)
            with self.assertRaises(ModelError):
                ForestModel.from_dict(d)

    def test_load_estimator_never_raises(self):
        est, err = load_estimator("/nonexistent/model.json", 0.03, 10)
        self.assertIsNone(est.model)
        self.assertIn("cannot open", err)


class TestManagerML(unittest.TestCase):
    def test_ml_mode_uses_model_after_warmup(self):
        model = ForestModel.from_dict(tiny_model_doc(leaf_values=(math.log(0.6), math.log(0.6))))
        est = MLTimeoutEstimator(model, rto_min=0.001, rto_max=10, warmup=8)
        tm = TimeoutManager("ml", rto_min=0.001, rto_max=10, ml=est, ml_interval=0.0)
        for i in range(20):
            tm.on_rtt_sample(i * 0.01, 0.05)
        self.assertEqual(tm.source, "ml")
        self.assertEqual(tm.fallbacks["warmup"], 7)          # samples 1..7
        self.assertGreater(tm.ml_decisions, 0)
        # 0.6 x RTO_conv, but never below the largest recent RTT (0.05)
        self.assertAlmostEqual(tm.base_rto, max(0.6 * tm.conv.rto, 0.05))

    def test_ml_mode_without_model_matches_conventional(self):
        est, _ = load_estimator(None, 0.03, 10)
        tm = TimeoutManager("ml", rto_min=0.03, rto_max=10, ml=est, ml_interval=0.0)
        for i in range(20):
            tm.on_rtt_sample(i * 0.01, 0.05 + 0.01 * (i % 3))
        self.assertEqual(tm.base_rto, tm.conv.rto)
        self.assertEqual(tm.ml_decisions, 0)
        self.assertEqual(tm.fallbacks["no_model"], 20)

    def test_rate_limited_recompute(self):
        model = ForestModel.from_dict(tiny_model_doc())
        est = MLTimeoutEstimator(model, rto_min=0.001, rto_max=10, warmup=1)
        tm = TimeoutManager("ml", rto_min=0.001, rto_max=10, ml=est, ml_interval=1.0)
        for i in range(10):
            tm.on_rtt_sample(i * 0.01, 0.05)
        self.assertEqual(tm.ml_decisions, 1)                 # once per second at most


if __name__ == "__main__":
    unittest.main()
