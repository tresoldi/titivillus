"""
Main module for the generation of stemma.
"""

# Import Python standard libraries
import random
from typing import Optional, Union

# Import other local modules
from .common import set_seeds


class Stemma:
    def __init__(self, dummy):
        self.dummy = dummy

    def __str__(self):
        return f"dummy stemma {self.dummy}"


# TODO: should we allow passing None to set_seeds(), to refresh the generators?
def random_stemma(seed: Optional[Union[str, int, float]] = None) -> Stemma:
    # Set the seed if it was provided; note that this will *not* pass `seed`
    # to `set_seeds()` if it is None, just leaving both generators (Python and numpy)
    # in the state they are
    if seed:
        set_seeds(seed)

    dummy_val = random.randint(0, 100)

    s = Stemma(dummy_val)

    return s
