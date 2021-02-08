"""
Common functions.
"""

# Import Python standard libraries
import hashlib
import random
from typing import Union, List

# Import 3rd party libraries
import numpy as np


# TODO: bring `unique_ids()` from langgenera?

def set_seeds(seed: Union[str, float, int]) -> None:
    """
    Set seeds globally from the user provided one.

    The function takes care of reproducibility and allows to use strings and
    floats as seed for `numpy` as well.

    :param seed: The seed for Python and numpy random number generators.
    """

    # Set seed for Python RNG
    random.seed(seed)

    # Allows using strings as numpy seeds, which only takes uint32 or arrays of uint32
    if isinstance(seed, (str, float)):
        np_seed = np.frombuffer(
            hashlib.sha256(str(seed).encode("utf-8")).digest(), dtype=np.uint32
        )
    else:
        np_seed = seed

    # Set the np set
    np.random.seed(np_seed)


def random_codex_name(used_names: List[str] = None) -> str:
    """
    Build and return a single random codex name.

    Random codex names follow practices of having the abbreviation for the name of a
    library, a description, a number, and occasionally additional information.
    """

    # TODO: expand with other libraries besides apophthegmata
    libraries = ["Athens", "Athos", "BeogMSPC", "BeogNBS", "Brux", "Cologn",
                 "DayrAlAbyad", "HML",
                 "HMS",
                 "LondAdd", "MilAmbr", "MoscGim", "MoscRGB", "MunSB", "ParCoisl",
                 "Sin", "SofiaNBKM", "StPeterBAN", "StPeterRNB", "Strasb", "Vat",
                 "Wien"]

    index = str(random.randint(1, 1500))

    # Use a loop to make sure there are no repeated names
    # TODO: while very improbable, this could technically get stuck in a loop if all
    #       possible names are exhausted, or just be unexpectedly slow; correct
    name_found = False
    while not name_found:
        # 3/4 of the time we add an alphabetic suffix, weighting towards the first ones
        if random.random() < 0.75:
            suffix = random.choices("ABCDEFGHIJ",
                                    weights=[512, 256, 128, 64, 32, 16, 8, 4, 2, 1])[0]
            name = f"{random.choice(libraries)}_{index}_{suffix}"
        else:
            name = f"{random.choice(libraries)}_{index}"

        # Check if it is a valid name
        if not used_names:
            name_found = True
        elif name not in used_names:
            name_found = True

    return name
