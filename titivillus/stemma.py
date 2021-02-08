"""
Main module for the generation of stemma.
"""

# TODO: random x,y position
# TODO: document weight

# Import Python standard libraries
import random
from typing import Optional, Union, List

# Import other local modules
from .common import set_seeds


class Stemma:
    def __init__(self, documents, source, ages):
        self.documents = documents
        self.source = source
        self.ages = ages

    def print(self):
        for d, c, a in zip(self.documents, self.source, self.ages):
            print(d, c, a)

    def __str__(self):
        return f"dummy stemma {len(self.documents)}"

    def to_dot(self):
        # see if the copy takes mostly from 0 or 1
        zeros = self.source[-1].count(0)
        ones = self.source[-1].count(1)
        print(zeros, ones)

# TODO: allow distribution, so that one root is more likely than the other
def _split_root_chars(char_list: list, num_roots: int) -> List[list]:
    if num_roots == 1:
        return [char_list]

    # Build a list with the same size of `char_list`, filling it with root indexes,
    # and distribute it
    roots = [[] for _ in range(num_roots)]
    root_idx = [random.randint(0, num_roots - 1) for _ in range(len(char_list))]
    for char, root in zip(char_list, root_idx):
        roots[root].append(char)

    return roots

def new_document(roots, max_char):
    # Pick a random index
    idx = random.randint( 0, len(roots)-1)

    # Make a copy of the characters and set them all as that source
    chars = roots[idx][:]
    source = [idx for _ in range(len(chars))]

    # random: delete a character
    # TODO: should have a distribution, "favoring" boundaries
    # TODO: should work in blocks
    if random.random() < 0.25:
        i = random.randint(0, len(chars)-1)
        chars[i] = None # set to None here so alignment is easier
        source[i] = (source[i], "delete")

    # unintentional move (which generalizes swap)
    # TODO: should favor closer moves
    # TODO: should work in blocks
    # TODO: decide what to do when `a` and `b` are the same
    if random.random() < 0.25:
        a = random.randint(0, len(chars)-1)
        b = random.randint(0, len(chars)-1-1)
        if chars[a] is not None:
            mchar, msour = chars[a], source[a]
            chars = chars[:a] + chars[a+1:]
            source = source[:a] + source[a+1:]
            chars = chars[:b] + [mchar] + chars[b:]
            source = source[:b] + [(msour, "move", a, b)] + source[b:]

    if random.random() > 0.25:
        i = random.randint(0, len(chars)-1)
        chars = chars[:i] + [max_char+1] + chars[i:]
        source = source[:i] + [("exnovo")] + source[i:]

    return chars, source

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
    roots = _split_root_chars(list(range(0, num_characters)), num_roots)

    # Define the source of each character in each root as `None`, that is, ex novo
    source = [[None for _ in doc] for doc in roots]

    # By definition, all roots here share the same age (=distance) of zero
    # TODO: we could allow for the newest root to be 0.0 and have others as negative
    ages = [0.0]*len(roots)

    # we need to collect the largest character index in use, in case a new
    # character will be created
    max_char = max([max(root_doc) for root_doc in roots])

    # Pick one random existing document and a modified one (a copy)
    # TODO: deal with multiple documents
    doc_chars, doc_source = new_document(roots, max_char)


    docs = roots
    docs.append(doc_chars)
    source.append(doc_source)
    ages.append(1.0)

    s = Stemma(docs, source, ages)

    return s
