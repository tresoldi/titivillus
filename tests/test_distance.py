#!/usr/bin/env python3

"""
test_distance
=============

Tests for the `distance` module of the `titivillus` package.
"""

# Import Python standard libraries
import unittest

# Import the library being tested
import titivillus


class TestDistance(unittest.TestCase):
    test1_seq_a = "kitten"
    test1_seq_b = "sitting"
    test2_seq_a = (1, 2, 3)
    test2_seq_b = [1, 2, 3]
    test3_seq_a = (1, 2, 3, 4, 5)
    test3_seq_b = (1, 2, 4, 3, 6, 7)

    def test_edit_distance(self):
        self.assertAlmostEqual(
            titivillus.distance.edit_distance(self.test1_seq_a, self.test1_seq_b),
            3.0,
            places=1)

        self.assertAlmostEqual(
            titivillus.distance.edit_distance(self.test2_seq_a, self.test2_seq_b),
            0.0,
            places=1)

        self.assertAlmostEqual(
            titivillus.distance.edit_distance(self.test3_seq_a, self.test3_seq_b),
            3.0,
            places=1)

    def test_jaccard_distance(self):
        self.assertAlmostEqual(
            titivillus.distance.jaccard_distance(self.test1_seq_a, self.test1_seq_b),
            0.7,
            places=1)

        self.assertAlmostEqual(
            titivillus.distance.jaccard_distance(self.test2_seq_a, self.test2_seq_b),
            0.0,
            places=1)

        self.assertAlmostEqual(
            titivillus.distance.jaccard_distance(self.test3_seq_a, self.test3_seq_b),
            0.428571,
            places=4)

    def test_mmcwapa_distance(self):
        self.assertAlmostEqual(
            titivillus.distance.mmcwpa_distance(self.test1_seq_a, self.test1_seq_b),
            0.538461,
            places=4)

        self.assertAlmostEqual(
            titivillus.distance.mmcwpa_distance(self.test2_seq_a, self.test2_seq_b),
            0.0,
            places=1)

        self.assertAlmostEqual(
            titivillus.distance.mmcwpa_distance(self.test3_seq_a, self.test3_seq_b),
            0.554638,
            places=4)


if __name__ == "__main__":
    # Explicitly creating and running a test suite allows to profile
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDistance)
    unittest.TextTestRunner(verbosity=2).run(suite)
