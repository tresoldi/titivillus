#!/usr/bin/env python3

"""
Performs community detection on preprocessed orthographic tabular data.
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def cluster_random(df: pd.DataFrame) -> List[int]:
    """
    Assign random cluster labels to data.
    """
    rows = df.shape[0]
    clusters = random.randint(2, int(rows / 2))

    return [random.randint(0, clusters) for _ in range(rows)]


def pca_decomposition(
    data: pd.DataFrame, n_components: Optional[Union[int, str]] = None
) -> pd.DataFrame:
    """
    Perform PCA decomposition on the data.
    """
    if n_components is None:
        n_components = min(len(data.columns), 10)

    try:
        pca_decomp = PCA(n_components=n_components)
        components = pca_decomp.fit_transform(data)
        colnames = [f"pc{idx + 1}" for idx in range(components.shape[1])]
        principal_df = pd.DataFrame(data=components, columns=colnames, index=data.index)
        logging.info(
            f"Explained variance ratio: {pca_decomp.explained_variance_ratio_}"
        )
    except Exception as e:
        logging.error(f"Error in PCA decomposition: {e}")
        raise

    return principal_df


def plot_clusters(principal: pd.DataFrame, labels: np.ndarray) -> None:
    """
    Plot the data with labels.
    """
    try:
        mss = principal.index.to_numpy()
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_xlabel("Principal Component 1", fontsize=15)
        ax.set_ylabel("Principal Component 2", fontsize=15)
        ax.grid()

        scatter = ax.scatter(
            principal.iloc[:, 0], principal.iloc[:, 1], c=labels, cmap="rainbow"
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)

        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting: {e}")
        raise


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a dataframe from a tab-delimited file.
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
    """
    parser = argparse.ArgumentParser(
        description="Prepare orthographic data for analysis."
    )
    parser.add_argument("input", type=str, help="The source JSON file for processing.")
    parser.add_argument(
        "-s",
        "--scale",
        choices=["none", "standard_nomean", "standard_mean"],
        default="none",
        help="Whether to perform preprocessing scaling and of which kind (default: none)",
    )
    parser.add_argument(
        "-d",
        "--decompose",
        choices=["none", "pca"],
        default="pca",
        help="Whether to perform decomposition and of which kind (default: pca)",
    )
    parser.add_argument(
        "-c",
        "--cluster",
        choices=["affinity", "random"],
        default="affinity",
        help="Whether to perform clustering and of which kind (default: affinity)",
    )
    args = parser.parse_args()
    return vars(args)


def main() -> None:
    """
    Script entry point.
    """
    args = parse_arguments()

    # Read tabular data as a pandas dataframe
    df = read_dataframe(args["input"])

    # Perform scaling if requested
    if args["scale"] == "standard_mean":
        df = pd.DataFrame(
            StandardScaler().fit_transform(df), index=df.index, columns=df.columns
        )
    elif args["scale"] == "standard_nomean":
        df = pd.DataFrame(
            StandardScaler(with_mean=False).fit_transform(df),
            index=df.index,
            columns=df.columns,
        )

    # Run decomposition if requested
    if args["decompose"] == "pca":
        df = pca_decomposition(df)

    # Run the clustering
    if args["cluster"] == "affinity":
        labels = cluster_affinity(df)
    elif args["cluster"] == "random":
        labels = cluster_random(df)

    # Log results
    logging.info("Data points and their cluster labels:")
    for index, label in zip(df.index, labels):
        logging.info(f"{index}: Cluster {label}")

    # Plot the results
    plot_clusters(df, labels)


if __name__ == "__main__":
    main()
