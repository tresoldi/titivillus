# __init__.py

"""
Titivillus __init__ file.
"""

# Version and general configuration for the package
__version__ = "0.1"
__author__ = "Tiago Tresoldi"
__email__ = "tiago.tresoldi@lingfil.uu.se"

# Build namespace
from .cluster import cluster_affinity, cluster_kmeans, cluster_hierarchical
from .common import read_dataframe, scale_data, save_to_csv
from .dimred import pca_decomposition
from .plot import plot_clusters
