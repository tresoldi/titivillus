# __init__.py

"""
Titivillus __init__ file.
"""

# Version and general configuration for the package
__version__ = "0.0.1"
__author__ = "Tiago Tresoldi"
__email__ = "tresoldi@gmail.com"

# Build namespace
from . import distance
from .codex import Codex, codex_distance, OriginCopy, OriginMove, OriginExNovo
from .common import set_seeds, collect_subseqs
from .ngrams import ngrams_iter
from .stemma import Stemma, random_stemma
