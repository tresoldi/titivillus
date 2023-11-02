#!/usr/bin/env python3

"""
Performs community detection on preprocessed orthographic tabular data.
"""

# Import Python standard libraries
import argparse
import logging
import random
from typing import Any, Dict, Union

# Import 3rd-party libraries
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cluster_affinity(df: pd.DataFrame) -> np.ndarray:
    """
    Perform clustering using the Affinity Propagation algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe for clustering.

    Returns
    -------
    np.ndarray
        An array of cluster labels.
    """
    try:
        clustering = AffinityPropagation(random_state=5).fit(df)
    except Exception as e:
        logging.error(f"Error in Affinity Propagation clustering: {e}")
        raise

    return clustering.labels_


def cluster_random(df: pd.DataFrame) -> list:
    """
    Assign random cluster labels to data.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe for clustering.

    Returns
    -------
    list
        A list of random cluster labels.
    """
    rows = len(df.index)
    clusters = random.randint(2, int(rows / 2))

    return [random.randint(0, clusters) for _ in range(rows)]


def pca(data: pd.DataFrame, n_components: Union[int, str]) -> pd.DataFrame:
    """
    Perform PCA decomposition on the data.

    Parameters
    ----------
    data : pd.DataFrame
        The input data for PCA.
    n_components : Union[int, str]
        The number of components to keep.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the principal components.
    """
    try:
        pca_decomp = PCA(n_components=n_components)
        components = pca_decomp.fit_transform(data)
        colnames = [f"pc{idx+1}" for idx in range(components.shape[1])]
        principal = pd.DataFrame(data=components, columns=colnames, index=data.index.values)
        logging.info(f"Explained variance ratio: {pca_decomp.explained_variance_ratio_}")
    except Exception as e:
        logging.error(f"Error in PCA decomposition: {e}")
        raise

    return principal


def plot(principal: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Plot the data with labels.

    Parameters
    ----------
    principal : pd.DataFrame
        The dataframe with the principal components.
    labels : np.ndarray
        The cluster labels for each data point.
    """
    try:
        mss = principal.index.values
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlabel("Principal Component 1", fontsize=15)
        ax.set_ylabel("Principal Component 2", fontsize=15)
        ax.grid()

        matrix = principal.to_numpy()
        colors = plt.cm.rainbow(np.linspace(0, 1, max(labels) + 1))
        for i in range(len(matrix)):
            x, y = matrix[i][0], matrix[i][1]
            ax.scatter(x, y, c=[colors[labels[i]]])
            ax.text(x * (1 + 0.03), y * (1 + 0.03), mss[i], fontsize=8)

        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting: {e}")
        raise


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a dataframe from a tab-delimited file.

    Parameters
    ----------
    filename : str
        The path to the file to read.

    Returns
    -------
    pd.DataFrame
        The dataframe read from the file.
    """
    try:
        df = pd.read_csv(filename, delimiter="\t", encoding="utf-8", index_col=0)
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
        raise
    except Exception as e:
        logging.error(f"Error reading the dataframe: {e}")
        raise

    return df


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.

    Returns
    -------
    Dict[str, Any]
        The command-line arguments as a dictionary.
    """
    parser = argparse.ArgumentParser(description="Prepare orthographic data for analysis.")
    parser.add_argument("input", type=str, help="The source JSON file for processing.")
    parser.add_argument(
        "-s", "--scale",
        type=str,
        choices=["none", "standard_nomean", "standard_mean"],
        default="none",
        help="Whether to perform preprocessing scaling and of which kind (default: none)"
    )
    parser.add_argument(
        "-d", "--decompose",
        type=str,
        choices=["none", "pca"],
        default="pca",
        help="Whether to perform decomposition and of which kind (default: pca)"
    )
    parser.add_argument(
        "-c", "--cluster",
        type=str,
        choices=["affinity", "random"],
        default="affinity",
        help="Whether to perform clustering and of which kind (default: affinity)"
    )
    return vars(parser.parse_args())


def main(arguments: Dict[str, Any]) -> None:
    """
    Script entry point.

    Parameters
    ----------
    arguments : Dict[str, Any]
        The dictionary of command-line arguments.
    """
    # Read tabular data as a pandas dataframe
    df = read_dataframe(arguments["input"])

    # Perform scaling if requested
    if arguments["scale"] == "standard_mean":
        df = StandardScaler().fit_transform(df)
    elif arguments["scale"] == "standard_nomean":
        df = StandardScaler(with_mean=False).fit_transform(df)

    # Run decomposition if requested
    if arguments["decompose"] == "pca":
        df = pca(df, 2)

    # Run the clustering
    if arguments["cluster"] == "affinity":
        labels = cluster_affinity(df)
    elif arguments["cluster"] == "random":
        labels = cluster_random(df)

    # Log results
    logging.info(f"Data points and their cluster labels: {list(zip(df.index.values, labels))}")

    # Plot results
    plot(df, labels)


if __name__ == "__main__":
    try:
        args = parse_arguments()
        main(args)
    except Exception as e:
        logging.error(f"An error occurred during the execution of the script: {e}")
