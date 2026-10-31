"""Statistics used in the experiment report."""

import unittest

from experiments.metrics import _ranks, bootstrap_ci, paired, wilcoxon_exact


class TestStatistics(unittest.TestCase):
    def test_wilcoxon_known_values(self):
        # all differences positive: only 2 of 2^n sign patterns are as extreme
        self.assertAlmostEqual(wilcoxon_exact([1, 2, 3, 4, 5]), 2 / 32)
        self.assertAlmostEqual(wilcoxon_exact(list(range(1, 11))), 2 / 1024)
        self.assertEqual(wilcoxon_exact([0, 0, 0]), 1.0)          # no evidence at all
        self.assertAlmostEqual(wilcoxon_exact([1, -1]), 1.0)

    def test_ranks_with_ties(self):
        self.assertEqual(_ranks([3, 1, 3, 2]), [3.5, 1.0, 3.5, 2.0])

    def test_bootstrap_ci_contains_mean(self):
        lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLess(lo, 3.0)
        self.assertGreater(hi, 3.0)

    def test_paired_summary(self):
        r = paired([10, 10, 10, 10], [11, 12, 11, 12])
        self.assertAlmostEqual(r["diff_mean"], 1.5)
        self.assertAlmostEqual(r["pct_change"], 15.0)
        self.assertEqual(r["n"], 4)


if __name__ == "__main__":
    unittest.main()
