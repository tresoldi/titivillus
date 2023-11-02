import pandas as pd
import numpy as np
import logging
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    AffinityPropagation,
    DBSCAN,
    SpectralClustering,
)
from typing import Callable, Dict

# Factory pattern setup
clustering_methods: Dict[str, Callable] = {}


def register_clustering(name: str) -> Callable:
    def inner(func: Callable) -> Callable:
        clustering_methods[name] = func
        return func

    return inner


@register_clustering("affinity")
def cluster_affinity(df: pd.DataFrame, **kwargs) -> np.ndarray:
    """
    Perform clustering using the Affinity Propagation algorithm.
    """
    clustering = AffinityPropagation(random_state=5, **kwargs).fit(df)
    return clustering.labels_


@register_clustering("kmeans")
def cluster_kmeans(df: pd.DataFrame, n_clusters: int = 8, **kwargs) -> np.ndarray:
    """
    Perform clustering using the K-Means algorithm.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=5, **kwargs).fit(df)
    logging.info(f"K-Means: Inertia: {kmeans.inertia_}")
    return kmeans.labels_


@register_clustering("hierarchical")
def cluster_hierarchical(df: pd.DataFrame, n_clusters: int = 8, **kwargs) -> np.ndarray:
    """
    Perform clustering using the Hierarchical clustering algorithm.
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, **kwargs).fit(df)
    return hierarchical.labels_


# Add new clustering methods as needed with decorators
@register_clustering("dbscan")
def cluster_dbscan(df: pd.DataFrame, **kwargs) -> np.ndarray:
    dbscan = DBSCAN(**kwargs).fit(df)
    return dbscan.labels_


@register_clustering("spectral")
def cluster_spectral(df: pd.DataFrame, n_clusters: int = 8, **kwargs) -> np.ndarray:
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=5, **kwargs).fit(
        df
    )
    return spectral.labels_


def perform_clustering(df: pd.DataFrame, method: str, **kwargs) -> np.ndarray:
    """
    Perform clustering using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to cluster.
    method : str
        The clustering method to use.

    Returns
    -------
    np.ndarray
        The cluster labels for each sample.
    """
    if method not in clustering_methods:
        raise ValueError(f"Clustering method '{method}' not recognized.")

    try:
        clustering_func = clustering_methods[method]
        return clustering_func(df, **kwargs)
    except Exception as e:
        logging.error(f"Error in '{method}' clustering: {e}")
        raise
