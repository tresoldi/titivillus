#!/usr/bin/env python3
"""
__main__.py

Module for command-line execution and generation of random networks.
"""

import logging
import argparse
from logging import DEBUG, INFO, WARNING, ERROR
from typing import Any, Dict

import titivillus

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    df = titivillus.read_dataframe(args["input"])

    # Perform scaling if not 'none'
    if args["scale"] != "none":
        df = titivillus.scale_data(df, args["scale"])

    # Run decomposition if requested
    if args["decompose"] == "pca":
        df = titivillus.pca_decomposition(df)

    # Run the clustering
    if args["cluster"] == "affinity":
        labels = titivillus.cluster_affinity(df)
    elif args["cluster"] == "kmeans":
        labels = titivillus.cluster_kmeans(df, n_clusters=args["clusters"])
    elif args["cluster"] == "hierarchical":
        labels = titivillus.cluster_hierarchical(df, n_clusters=args["clusters"])

    # Log results
    logging.info("Data points and their cluster labels:")
    for index, label in zip(df.index, labels):
        logging.info(f"{index}: Cluster {label}")

    # Plot the results in 2D
    titivillus.plot_clusters(df, labels)

    # Additional plots if PCA has at least 3 components
    if df.shape[1] >= 3:
        titivillus.plot_clusters(df, labels, plot_type="3d")  # Plot in 3D

    # Plot heatmap if appropriate (note: best for smaller number of features due to readability)
    if (
        df.shape[1] <= 20
    ):  # Adjust this threshold based on your dataset and readability preferences
        titivillus.plot_clusters(
            df, labels, plot_type="heatmap"
        )  # Plot heatmap of similarity matrix

    # Save results to CSV if output file is specified
    if args["output"]:
        titivillus.save_to_csv(df, labels, args["output"])


if __name__ == "__main__":
    main()
