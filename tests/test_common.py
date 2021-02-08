#!/usr/bin/env python3

"""
test_common
===========

Tests for the `common` module of the `titivillus` package.
"""

# Import Python standard libraries
import random
import unittest

# Import the library being tested
import titivillus


class TestCommon(unittest.TestCase):

    # TODO: can we actually compare across different environments?
    def test_set_seeds(self):
        # As the actual numbers can vary between platforms, we cannot test against the
        # actual results of random calls, but only initialize twice and verify that
        # the same calls generate the same numbers.

        # Test with integers
        titivillus.set_seeds(42)
        num_a = random.randint(0, 100)
        titivillus.set_seeds(42)
        num_b = random.randint(0, 100)
        assert num_a == num_b


if __name__ == "__main__":
    # Explicitly creating and running a test suite allows to profile
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCommon)
    unittest.TextTestRunner(verbosity=2).run(suite)
