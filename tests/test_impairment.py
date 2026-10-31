"""Network simulator: determinism, rates, bursts, profiles."""

import unittest

from protocol.packet import MsgType
from simulation.impairment import FWD, REV, Impairment
from simulation.profiles import PROFILES, Burst, Phase, Profile, get_profile


class TestDeterminism(unittest.TestCase):
    def test_same_key_same_decision(self):
        a = Impairment(get_profile("dup_reorder"), seed=7)
        b = Impairment(get_profile("dup_reorder"), seed=7)
        for seq in range(500):
            self.assertEqual(a.decide(FWD, MsgType.DATA, seq, 0, 1.0),
                             b.decide(FWD, MsgType.DATA, seq, 0, 1.0))

    def test_order_independence(self):
        # The fate of a copy does not depend on what was decided before it:
        # this is what makes paired comparisons of two timeout strategies fair.
        imp = Impairment(get_profile("lossy"), seed=3)
        forward = [imp.decide(FWD, MsgType.DATA, s, 0, 0.5) for s in range(300)]
        imp2 = Impairment(get_profile("lossy"), seed=3)
        backward = [imp2.decide(FWD, MsgType.DATA, s, 0, 0.5) for s in reversed(range(300))]
        self.assertEqual(forward, list(reversed(backward)))

    def test_different_copy_or_seed_differs(self):
        imp = Impairment(get_profile("lossy"), seed=1)
        d0 = [imp.decide(FWD, MsgType.DATA, s, 0, 0.0).delays for s in range(100)]
        d1 = [imp.decide(FWD, MsgType.DATA, s, 1, 0.0).delays for s in range(100)]
        other = Impairment(get_profile("lossy"), seed=2)
        d2 = [other.decide(FWD, MsgType.DATA, s, 0, 0.0).delays for s in range(100)]
        self.assertNotEqual(d0, d1)
        self.assertNotEqual(d0, d2)


class TestRates(unittest.TestCase):
    def test_loss_rates_match_profile(self):
        imp = Impairment(get_profile("lossy"), seed=11)
        n = 40000
        fwd = sum(imp.decide(FWD, MsgType.DATA, s, 0, 1.0).drop for s in range(n)) / n
        rev = sum(imp.decide(REV, MsgType.ACK, s, 0, 1.0).drop for s in range(n)) / n
        self.assertAlmostEqual(fwd, 0.05, delta=0.005)
        self.assertAlmostEqual(rev, 0.02, delta=0.004)

    def test_delay_within_jitter(self):
        imp = Impairment(get_profile("stable"), seed=5)
        delays = [imp.decide(FWD, MsgType.DATA, s, 0, 0.0).delays[0] for s in range(2000)]
        self.assertGreaterEqual(min(delays), 0.009 - 1e-9)
        self.assertLessEqual(max(delays), 0.011 + 1e-9)

    def test_duplication_and_corruption_happen(self):
        imp = Impairment(get_profile("dup_reorder"), seed=5)
        ds = [imp.decide(FWD, MsgType.DATA, s, 0, 0.0) for s in range(20000)]
        delivered = [d for d in ds if not d.drop]
        dup = sum(d.duplicated for d in delivered) / len(delivered)
        cor = sum(d.corrupt for d in delivered) / len(delivered)
        self.assertAlmostEqual(dup, 0.02, delta=0.005)
        self.assertAlmostEqual(cor, 0.01, delta=0.004)
        self.assertTrue(all(len(d.delays) == 2 for d in delivered if d.duplicated))

    def test_pareto_jitter_has_heavy_tail(self):
        imp = Impairment(get_profile("jitter_heavy"), seed=5)
        delays = sorted(d.delays[0] for d in
                        (imp.decide(FWD, MsgType.DATA, s, 0, 0.0) for s in range(20000)) if not d.drop)
        median, p999 = delays[len(delays) // 2], delays[int(len(delays) * 0.999)]
        self.assertGreater(p999 - 0.030, 10 * (median - 0.030))


class TestBursts(unittest.TestCase):
    def test_schedule_reproducible_and_loss_concentrated(self):
        prof = get_profile("bursty_loss")
        a, b = Impairment(prof, 9), Impairment(prof, 9)
        self.assertEqual(a.bad_periods, b.bad_periods)
        self.assertNotEqual(a.bad_periods, Impairment(prof, 10).bad_periods)
        self.assertGreater(len(a.bad_periods), 10)
        start, end = a.bad_periods[0]
        self.assertTrue(a.in_burst((start + end) / 2))
        self.assertFalse(a.in_burst(start - 1e-6))
        inside = [a.decide(FWD, MsgType.DATA, s, 0, (start + end) / 2).drop for s in range(4000)]
        self.assertAlmostEqual(sum(inside) / len(inside), 0.5, delta=0.03)


class TestProfiles(unittest.TestCase):
    def test_ramp_and_cycle(self):
        prof = Profile("t", "", [Phase(1.0, 10.0), Phase(2.0, 10.0, delay_end_ms=30.0)])
        self.assertEqual(prof.phase_at(0.5)[1], 10.0)
        self.assertAlmostEqual(prof.phase_at(2.0)[1], 20.0)    # halfway through the ramp
        self.assertAlmostEqual(prof.phase_at(3.5)[1], 10.0)    # second cycle
        self.assertEqual(prof.cycle, 3.0)

    def test_all_profiles_valid(self):
        for name, p in PROFILES.items():
            self.assertGreater(p.cycle, 0, name)
            imp = Impairment(p, 1)
            for t in (0.0, 1.3, 7.7):
                imp.decide(FWD, MsgType.DATA, 1, 0, t)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            get_profile("nope")


if __name__ == "__main__":
    unittest.main()
