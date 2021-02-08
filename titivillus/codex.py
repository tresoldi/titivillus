"""
Module defining a Codex dataclass.
"""

import random
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from .common import random_codex_name


@dataclass
class Codex:
    """
    Class for representing a single document in a stemma.

    `chars` is the ordered tuple of (phylogenetic) characters in the codex, represented
    by ints.
    `origins` if a tuple of tuples, each entry related to the entries of `char` in
    order, with a string indicating the type of origin for the character (like
    copy, innovation and move) and the second an integer carrying complementary
    information.
    `age` is the relative age of the codex, with the root (or, at least, the newest
    root) set by definition at 0.0.
    `weight` is the relative importance of the codex, with higher values indicating
    manuscripts that are more likely to influence in the tradition; if not provided,
    a random number between 0.0 and 100.0 will be generated
    `name` is a label for the codex, which will be generated automatically in
    post-initialization if necessary.
    """
    chars: Tuple[int, ...]
    origins: Tuple[Tuple[str, Optional[int]], ...]
    age: float
    weight: float = None
    name: str = None

    def __post_init__(self):
        """
        Perform post initialization operations, mostly setting values not provided.

        The RNG is used in the state it is found (i.e., no seeding).
        """

        # Generate a random manuscript name, if none was provided
        if not self.name:
            self.name = random_codex_name()

        # Generate a random weight, if none was provided; we try to guarantee that
        # the precision is of only two decimal places
        if not self.weight:
            self.weight = int(random.random() * 10000) / 100.0

        # Run checks
        # TODO: add check for types of origins
        # TODO: check or force value ranges?
        if len(self.chars) != len(self.origins):
            raise ValueError("Mismatch in length between `chars` and `origins`.")


# TODO: allow some normalization?
def edit_distance(seq1, seq2):
    # For all `x` and `y`, matrix[i, j] will hold the Levenshtein distance between the
    # first `x` characters of `seq1` and the first `y` characters of `seq2`; the
    # starting matrix is a zeroed one
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))

    # Source prefixes can be transformed into empty sequences by dropping all chars
    for x in range(size_x):
        matrix[x, 0] = x

    # Target prefixes can be reached from empty sequence prefix by inserting every char
    for y in range(size_y):
        matrix[0, y] = y

    # Main loop, with the implied substitution cost
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # deletion
                    matrix[x - 1, y - 1],  # insertion
                    matrix[x, y - 1] + 1  # substitution
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,  # deletion
                    matrix[x - 1, y - 1] + 1,  # insertion
                    matrix[x, y - 1] + 1  # substitution
                )

    return (matrix[size_x - 1, size_y - 1])


def jaccard_distance(seq1, seq2):
    intersection = len(set(seq1).intersection(seq2))
    union = len(seq1) + len(seq2) - intersection
    return 1.0 - (float(intersection) / union)


def codex_distance(codex1: Codex, codex2: Codex, method: str = "edit") -> float:
    """
    Computes the distance between two codices.
    """

    if method == "edit":
        return edit_distance(codex1.chars, codex2.chars)
    elif method == "jaccard":
        return jaccard_distance(codex1.chars, codex2.chars)

    raise ValueError("Unsupported distance method.")
