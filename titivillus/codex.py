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
# TODO: custom costs?
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

    return matrix[size_x - 1, size_y - 1]


def jaccard_distance(seq1, seq2):
    intersection = len(set(seq1).intersection(seq2))
    union = len(seq1) + len(seq2) - intersection
    return 1.0 - (float(intersection) / union)


def _mmcwpa(f_x, f_y, ssnc):
    """
    An implementation of the Modified Moving Contracting
    Window Pattern Algorithm (MMCWPA) to calculate string
    similarity, returns a list of non-overlapping,
    non-contiguous fields Fx, a list of non-overlapping,
    non-contiguous fields Fy, and a number indicating the
    Sum of the Square of the Number of the same
    Characters. This function is intended to be "private",
    called from the "public" stringcomp() function below.
    @param f_x: A C{list} of C{strings}.
    @param f_y: A C{list} of C{strings}.
    @param ssnc: A C{float}.
    @return: A C{list} of C{strings} with non-overlapping,
             non-contiguous subfields for Fx, a C{list} of
             C{strings} with non-overlapping,
             non-contiguous subfields for Fy, and a C{float}
             with the value of the SSNC collected so far.
    @rtype: C{list} of C{strings}, C{list} of strings,
            C{float}
    """

    # TODO: properly rewrite
    def _find_in_list(hay, needle):
        # Cache `needle` length and have it as a tuple already
        len_needle = len(needle)
        t_needle = tuple(needle)

        # Iterate over all sub-lists (or sub-tuples) of the correct length and check
        # for matches
        for i in range(len(hay) - len_needle + 1):
            if tuple(hay[i:i + len_needle]) == t_needle:
                return i

        return None

    # the boolean value indicating if a total or partial
    # match was found between subfields Fx and Fy; when
    # a match is found, the variable is used to cascade
    # out of the loops of the function
    match = False

    # the variables where to store the new collections of
    # subfields, if any match is found; if these values
    # are not changed and the empty lists are returned,
    # stringcomp() will break the loop of comparison,
    # calculate the similarity ratio and return its value
    new_f_x, new_f_y = [], []

    # search patterns in all subfields of Fx; the index of
    # the subfield in the list is used for upgrading the
    # list, if a pattern is a found
    for idx_x, sf_x in enumerate(f_x):
        # 'length' stores the length of the sliding window,
        # from full length to a single character
        for length in range(len(sf_x), 0, -1):
            # 'i' stores the starting index of the sliding
            # window in Fx
            for i in range(len(sf_x) - length + 1):
                # extract the pattern for matching
                pattern = sf_x[i:i + length]

                # look for the pattern in Fy
                for idx_y, sf_y in enumerate(f_y):
                    # 'j' stores the starting index in Fy; the
                    # Python find() function returns -1 if there
                    # is no match
                    j = _find_in_list(sf_y, pattern)
                    if j is not None:
                        # the pattern was found; set 'new_fx' and
                        # 'new_fy' to version of 'fx' and 'fy' with
                        # the patterns removed, update the SSNC and
                        # set 'match' as True, in order to cascade
                        # out of the loops
                        tmp_x = [sf_x[:i], sf_x[i + length:]]
                        tmp_y = [sf_y[:j], sf_y[j + length:]]
                        new_f_x = f_x[:idx_x] + tmp_x + f_x[idx_x + 1:]
                        new_f_y = f_y[:idx_y] + tmp_y + f_y[idx_y + 1:]

                        ssnc += (2 * length) ** 2

                        match = True

                        break

                    # if the current match was found, end search
                    if match:
                        break

                # if a match was found, end the sliding window
                if match:
                    break

            # if a match was found, end Fx subfield enumeration
            if match:
                break

        # remove any empty subfields due to pattern removal
        new_f_x = [sf for sf in new_f_x if sf]
        new_f_y = [sf for sf in new_f_y if sf]

        return new_f_x, new_f_y, ssnc


# TODO: rename str to seq
def mmcwpa_distance(str_x, str_y):
    len_x, len_y = len(str_x), len(str_y)

    f_x, f_y = [str_x], [str_y]

    ssnc = 0.0
    while True:
        f_x, f_y, ssnc = _mmcwpa(f_x, f_y, ssnc)

        if len(f_x) == 0 or len(f_y) == 0:
            break

    return 1.0 - ((ssnc / ((len_x + len_y) ** 2.)) ** 0.5)


def codex_distance(codex1: Codex, codex2: Codex, method: str = "edit") -> float:
    """
    Computes the distance between two codices.
    """

    if method == "edit":
        return edit_distance(codex1.chars, codex2.chars)
    elif method == "jaccard":
        return jaccard_distance(codex1.chars, codex2.chars)
    elif method == "mmcwpa":
        return mmcwpa_distance(codex1.chars, codex2.chars)

    raise ValueError("Unsupported distance method.")
