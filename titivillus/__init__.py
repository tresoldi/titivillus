# __init__.py

"""
Titivillus __init__ file.
"""

# Version and general configuration for the package
__version__ = "0.0.1"
__author__ = "Tiago Tresoldi"
__email__ = "tresoldi@gmail.com"

# Build namespace
from .codex import Codex, codex_distance
from .common import set_seeds
from .stemma import Stemma, random_stemma
from . import distance