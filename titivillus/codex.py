"""
Module defining a Codex dataclass.
"""

import random
from dataclasses import dataclass
from typing import Tuple

from .common import random_codex_name
from .distance import edit_distance, jaccard_distance, mmcwpa_distance


class Origin:
    def __init__(self):
        pass

    @property
    def source(self):
        """
        Property with the index of the source codex, or None if not applicable.
        """
        return None


class OriginExNovo(Origin):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "[exnovo]"


class OriginCopy(Origin):
    def __init__(self, source_idx):
        super().__init__()
        self.source_idx = source_idx

    @property
    def source(self):
        return self.source_idx

    def __repr__(self):
        return f"[copy #{self.source_idx}]"


class OriginMove(Origin):
    def __init__(self, source_idx, start_pos):
        super().__init__()
        self.source_idx = source_idx
        self.start_pos = start_pos

    @property
    def source(self):
        return self.source_idx

    def __repr__(self):
        return f"[copy #{self.source_idx}, move {self.start_pos}]"


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
    origins: Tuple[Origin, ...]
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
