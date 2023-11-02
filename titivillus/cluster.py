import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation


def cluster_affinity(df: pd.DataFrame) -> np.ndarray:
    """
    Perform clustering using the Affinity Propagation algorithm.
    """
    try:
        clustering = AffinityPropagation(random_state=5).fit(df)
    except Exception as e:
        logging.error(f"Error in Affinity Propagation clustering: {e}")
        raise

    return clustering.labels_


def cluster_kmeans(df: pd.DataFrame, n_clusters: int = 8) -> np.ndarray:
    """
    Perform clustering using the K-Means algorithm.
    """
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=5).fit(df)
    except Exception as e:
        logging.error(f"Error in K-Means clustering: {e}")
        raise

    return kmeans.labels_


def cluster_hierarchical(df: pd.DataFrame, n_clusters: int = 8) -> np.ndarray:
    """
    Perform clustering using the Hierarchical clustering algorithm.
    """
    try:
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters).fit(df)
    except Exception as e:
        logging.error(f"Error in Hierarchical clustering: {e}")
        raise

    return hierarchical.labels_
