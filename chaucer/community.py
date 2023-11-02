#!/usr/bin/env python3

"""
Performs community detection on preprocessed orthographic tabular data.
"""

import argparse
import logging
from logging import DEBUG, INFO, WARNING, ERROR
from typing import Any, Dict, Optional, Union

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap plotting

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


def plot_clusters(
    principal: pd.DataFrame, labels: np.ndarray, plot_type: str = "2d"
) -> None:
    """
    Plot the data with labels.
    Supports 2D, 3D, and heatmap plots based on 'plot_type'.
    """
    try:
        # For 3D plotting
        if plot_type == "3d":
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("Principal Component 1", fontsize=15)
            ax.set_ylabel("Principal Component 2", fontsize=15)
            ax.set_zlabel("Principal Component 3", fontsize=15)
            ax.scatter(
                principal.iloc[:, 0],
                principal.iloc[:, 1],
                principal.iloc[:, 2],
                c=labels,
                cmap="rainbow",
            )

        # For heatmap plotting
        elif plot_type == "heatmap":
            similarity_matrix = np.corrcoef(principal.transpose())
            sns.heatmap(similarity_matrix, cmap="coolwarm")
            plt.title("Heatmap of Similarity Matrix")

        # Default 2D plotting
        else:
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.set_xlabel("Principal Component 1", fontsize=15)
            ax.set_ylabel("Principal Component 2", fontsize=15)
            scatter = ax.scatter(
                principal.iloc[:, 0], principal.iloc[:, 1], c=labels, cmap="rainbow"
            )
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)

        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting: {e}")
        raise


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


def scale_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Scale the data using the specified method.
    """
    scalers = {
        "standard": StandardScaler(),
        "standard_nomean": StandardScaler(with_mean=False),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "l2_norm": Normalizer(norm="l2"),
    }

    if method not in scalers:
        logging.error(f"Scaling method '{method}' not recognized.")
        raise ValueError(f"Scaling method '{method}' not recognized.")

    try:
        scaler = scalers[method]
        scaled_df = pd.DataFrame(
            scaler.fit_transform(df), index=df.index, columns=df.columns
        )
        logging.info(f"Data scaled using {method} method.")
        return scaled_df
    except Exception as e:
        logging.error(f"Error scaling data: {e}")
        raise


def save_to_csv(data: pd.DataFrame, labels: np.ndarray, output_file: str) -> None:
    """
    Save the data along with its cluster labels to a CSV file.
    """
    try:
        # Add the labels to the dataframe
        output_df = data.assign(Cluster=labels)
        output_df.to_csv(output_file, index=True)
        logging.info(f"Cluster labels saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        raise


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare orthographic data for analysis."
    )
    parser.add_argument("input", type=str, help="The source JSON file for processing.")
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
        choices=["affinity", "kmeans", "hierarchical"],
        default="affinity",
        help="Which clustering algorithm to use (default: affinity)",
    )
    parser.add_argument(
        "-k",
        "--clusters",
        type=int,
        default=8,
        help="Number of clusters for K-Means and Hierarchical (default: 8)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="clustering_output.csv",
        help="The output CSV file path for saving the clustering results (default: clustering_output.csv)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        choices=["none", "standard", "standard_nomean", "minmax", "robust", "l2_norm"],
        default="none",
        help="The scaling method to apply to the data (default: none)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Verbosity level: 0=WARNING, 1=INFO, 2=DEBUG, 3=ERROR (default: 1)",
    )

    args = parser.parse_args()
    return vars(args)


def main() -> None:
    """
    Script entry point.
    """
    args = parse_arguments()

    # Set logging level based on verbosity
    if args["verbosity"] == 0:
        level = WARNING
    elif args["verbosity"] == 2:
        level = DEBUG
    elif args["verbosity"] == 3:
        level = ERROR
    else:
        level = INFO

    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Read tabular data as a pandas dataframe
    df = read_dataframe(args["input"])

    # Perform scaling if not 'none'
    if args["scale"] != "none":
        df = scale_data(df, args["scale"])

    # Run decomposition if requested
    if args["decompose"] == "pca":
        df = pca_decomposition(df)

    # Run the clustering
    if args["cluster"] == "affinity":
        labels = cluster_affinity(df)
    elif args["cluster"] == "kmeans":
        labels = cluster_kmeans(df, n_clusters=args["clusters"])
    elif args["cluster"] == "hierarchical":
        labels = cluster_hierarchical(df, n_clusters=args["clusters"])

    # Log results
    logging.info("Data points and their cluster labels:")
    for index, label in zip(df.index, labels):
        logging.info(f"{index}: Cluster {label}")

    # Plot the results in 2D
    plot_clusters(df, labels)

    # Additional plots if PCA has at least 3 components
    if df.shape[1] >= 3:
        plot_clusters(df, labels, plot_type="3d")  # Plot in 3D

    # Plot heatmap if appropriate (note: best for smaller number of features due to readability)
    if (
        df.shape[1] <= 20
    ):  # Adjust this threshold based on your dataset and readability preferences
        plot_clusters(
            df, labels, plot_type="heatmap"
        )  # Plot heatmap of similarity matrix

    # Save results to CSV if output file is specified
    if args["output"]:
        save_to_csv(df, labels, args["output"])


if __name__ == "__main__":
    main()
