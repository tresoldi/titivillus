#!/usr/bin/env python3

"""
test_codex
==========

Tests for the `codex` module of the `titivillus` package, especially the
`Codex` class.
"""

# Import Python standard libraries
import unittest

# Import the library being tested
import titivillus


class TestCodex(unittest.TestCase):
    def test_create(self):
        """
        Tests for Codex creation.
        """

        # Test a proper codex creation, with and without optional arguments
        codex = titivillus.Codex(
            (1, 2, 3),  # chars
            (("copy", 1), ("copy", 1), ("copy", 1)),  # origins
            1.0,  # age
        )

        codex = titivillus.Codex(
            (1, 2, 3),  # chars
            (("copy", 1), ("copy", 1), ("copy", 1)),  # origins
            1.0,  # age
            3.0,  # weight
            "Man_Name",  # name
        )

        # Test various exceptions for missing or wrong information
        with self.assertRaises(TypeError):
            codex = titivillus.Codex()

    # TODO: should equality consider the name? perhaps only chars, or a distance
    #       measure for those?
    def test_equality(self):
        """
        Test equality methods.
        """

        chars = (1, 2, 3)
        origins = ("copy", 1), ("copy", 1), ("copy", 1)
        age = 1.0
        weight = 1.0
        name = "dummy_name"

        codex1 = titivillus.Codex(chars, origins, age, weight, name)
        codex2 = titivillus.Codex(chars, origins, age, weight, name)
        codex3 = titivillus.Codex(chars, origins, 2.0, weight, name)

        assert codex1 == codex2
        assert codex1 != codex3


if __name__ == "__main__":
    # Explicitly creating and running a test suite allows to profile
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCodex)
    unittest.TextTestRunner(verbosity=2).run(suite)
