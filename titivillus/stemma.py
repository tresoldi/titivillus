"""
Main module for the generation of stemma.
"""

# TODO: random x,y position
# TODO: document weight

# Import Python standard libraries
import random
from itertools import chain
from typing import Optional, Union, List

import networkx as nx

from .codex import Codex, OriginExNovo, OriginMove, OriginCopy, Origin

# Import other local modules
from .common import set_seeds


# TODO: accept empty list of codices?
class Stemma:
    def __init__(self, codices: List[Codex]):
        self.codices = codices

    def __str__(self):
        return f"Stemma of {len(self.codices)} codices and {len(self.charset)} chars."

    def as_graph(self):
        DG = nx.DiGraph()
        for codex in self.codices:
            DG.add_node(codex.name)

        for codex in self.codices:
            sources = [origin.source for origin in codex.origins]
            sources = [idx for idx in sources if idx is not None]

            for s in sources:
                DG.add_edge(self.codices[s].name, codex.name)

        return DG

    # TODO: could return a counter (or have a method for that) and have the charset
    #       from the keys
    @property
    def charset(self) -> set:
        """
        Return a set with all the characters used in the stemma.
        """

        return set(chain.from_iterable([codex.chars for codex in self.codices]))


# TODO: allow distribution, so that one root is more likely than the other
def _split_root_chars(char_list: list, num_roots: int) -> List[tuple]:
    if num_roots == 1:
        return [tuple(char_list)]

    # Build a list with the same size of `char_list`, filling it with root indexes,
    # and distribute it
    roots = [[] for _ in range(num_roots)]
    root_idx = [random.randint(0, num_roots - 1) for _ in range(len(char_list))]
    for char, root in zip(char_list, root_idx):
        roots[root].append(char)

    return [tuple(r) for r in roots]


# TODO: move to codex.py? to some "random generator"?
def random_codex(stemma: Stemma) -> Codex:
    # Pick a random index in the stemma to be the main source
    # TODO: pick by using (x,y) and weight
    cidx = random.randint(0, len(stemma.codices) - 1)

    # Grab a copy of the characters in the selected codex, and generate a list
    # of sources for all of them. `chars` and `origin` are at first lists, to make the
    # manipulation easier
    chars = list(stemma.codices[cidx].chars)
    origin: List[Origin] = [OriginCopy(cidx) for _ in chars]

    # random: delete a character
    # TODO: should have a distribution, "favoring" boundaries
    # TODO: should work in blocks
    if random.random() < 0.25:
        char_idx = random.randint(0, len(chars) - 1)
        chars.pop(char_idx)
        origin.pop(char_idx)

    # unintentional move (which generalizes swap)
    # TODO: should favor closer moves
    # TODO: should work in blocks
    # TODO: decide what to do when `a` and `b` are the same
    if random.random() < 0.25:
        # Random select and pop a character and its origin
        a = random.randint(0, len(chars) - 1)
        m_char = chars.pop(a)
        m_origin = origin.pop(a)

        # Randomly select a destination: note that, at this point, `chars` and `origin`
        # already have had their elements removed. As for the origin, we need to
        # build a new element that involves the movement and the origin.
        # TODO: should have a movement element even if it happens to stay in the same
        #       position?
        b = random.randint(0, len(chars) - 1)
        chars.insert(b, m_char)
        origin.insert(b, OriginMove(m_origin.source, a))

    # innovation
    if random.random() < 0.25:
        max_char = max(stemma.charset)
        idx = random.randint(0, len(chars) - 1)

        chars.insert(idx, max_char)
        origin.insert(idx, OriginExNovo())

    return Codex(tuple(chars), tuple(origin), stemma.codices[cidx].age + 1.0)


# TODO: should we allow passing None to set_seeds(), to refresh the generators?
# TODO: number of roots should come from a distribution (Poisson?), also considering
#       the size of each manuscript (number of characters)
# TODO: num_characters should better be defined as number of initial characters (sum
#       of all roots), and not the total number as it would make it harder to have
#       new ones popping in without setting new roots
def random_stemma(seed: Optional[Union[str, int, float]] = None, **kwargs) -> Stemma:
    # Parse additional arguments, checking types and setting defaults
    num_characters: int = kwargs.get("num_characters", 20)
    num_roots: int = kwargs.get("num_roots", 1)

    # Set the seed if it was provided; note that this will *not* pass `seed`
    # to `set_seeds()` if it is None, just leaving both generators (Python and numpy)
    # in the state they are
    if seed:
        set_seeds(seed)

    # Generate the distribution of characters, and their orders, in the roots. By
    # definition, a character cannot be found in more than one root -- it can be
    # a single, separate root of itself, shared by other roots, but it cannot
    # be shared.
    roots_chars = _split_root_chars(list(range(0, num_characters)), num_roots)

    # By definition, all roots here share the same age (=distance) of zero
    # TODO: we could allow for the newest root to be 0.0 and have others as negative
    codices = [
        Codex(charset, tuple([OriginExNovo() for _ in range(len(charset))]), 0.0)
        for charset in roots_chars
    ]

    stemma = Stemma(codices)

    for i in range(8):
        cdx = random_codex(stemma)
        stemma.codices.append(cdx)
        print(stemma)

    return stemma
