#!/usr/bin/env python3
"""
This is the main module for the `titivillus` package. It provides command-line interfaces
to process input data, perform scaling and decomposition, execute clustering algorithms,
and save the results to a file.
"""

# Import standard modules
import argparse
import logging
import os
import sys

# Import third-party modules
import pandas as pd

# Import local modules
import titivillus

# Constants
DEFAULT_OUTPUT_FILENAME = "clustering_output.csv"
DEFAULT_NUM_CLUSTERS = 8
MAX_FEATURES_FOR_HEATMAP = 20
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Verbosity level dictionary
VERBOSITY_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
    3: logging.ERROR,
}


def configure_logging(verbosity: int) -> None:
    """
    Configure the logging level based on the verbosity option.

    Parameters
    ----------
    verbosity : int
        The verbosity level; higher numbers enable more verbose output.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=VERBOSITY_LEVEL.get(verbosity, logging.INFO), format=LOG_FORMAT
    )


def read_data(input_file: str) -> pd.DataFrame:
    """
    Read data from a specified input file.

    Parameters
    ----------
    input_file : str
        The path to the input file to be processed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed data.

    Raises
    ------
    SystemExit
        If the file cannot be found or read.
    """
    try:
        return titivillus.read_dataframe(input_file)
    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
        sys.exit(1)
    except ValueError as e:
        logging.error(f"Error reading input file: {e}")
        sys.exit(1)


def scale_and_decompose(df: pd.DataFrame, scale: str, decompose: str) -> pd.DataFrame:
    """
    Apply scaling and decomposition to the data if required.

    Parameters
    ----------
    df : pd.DataFrame
        The data to be scaled and/or decomposed.
    scale : str
        The scaling method to apply.
    decompose : str
        The decomposition method to apply.

    Returns
    -------
    pd.DataFrame
        The scaled and/or decomposed data.
    """
    if scale != "none":
        df = titivillus.scale_data(df, scale)
    if decompose == "pca":
        df = titivillus.pca_decomposition(df)
    return df


def perform_clustering(
    df: pd.DataFrame, cluster_method: str, num_clusters: int
) -> pd.Series:
    """
    Perform clustering on the data using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        The data to cluster.
    cluster_method : str
        The clustering algorithm to use.
    num_clusters : int
        The number of clusters to form.

    Returns
    -------
    pd.Series
        The cluster labels for each data point.

    Raises
    ------
    ValueError
        If an unknown clustering method is provided.
    """
    if cluster_method == "affinity":
        labels = titivillus.cluster_affinity(df)
    elif cluster_method == "kmeans":
        labels = titivillus.cluster_kmeans(df, n_clusters=num_clusters)
    elif cluster_method == "hierarchical":
        labels = titivillus.cluster_hierarchical(df, n_clusters=num_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {cluster_method}")
    return labels


def generate_plots(df: pd.DataFrame, labels: pd.Series, output_prefix: str) -> None:
    """
    Generate and save plots for the clustered data to disk.

    Parameters
    ----------
    df : pd.DataFrame
        The data to plot.
    labels : pd.Series
        The cluster labels for each data point.
    output_prefix : str
        The prefix for the filenames of the output plots.

    Returns
    -------
    None
    """
    base_filename = f"{output_prefix}.2dpca.png"
    titivillus.plot_clusters(df, labels, save_path=base_filename)

    if df.shape[1] >= 3:
        base_filename = f"{output_prefix}.3dpca.png"
        titivillus.plot_clusters(df, labels, plot_type="3d", save_path=base_filename)

    if df.shape[1] <= MAX_FEATURES_FOR_HEATMAP:
        base_filename = f"{output_prefix}.heatmap.png"
        titivillus.plot_clusters(
            df, labels, plot_type="heatmap", save_path=base_filename
        )


def save_results(df: pd.DataFrame, labels: pd.Series, output_prefix: str) -> None:
    """
    Save the clustering results to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The data to save.
    labels : pd.Series
        The cluster labels for each data point.
    output_prefix : str
        The prefix for the filenames of the output files.

    Returns
    -------
    None
    """
    output_path = f"{output_prefix}.clusters.csv"
    try:
        titivillus.save_to_csv(df, labels, output_path)
    except Exception as e:
        logging.error(f"Failed to save results to '{output_path}': {e}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
        An object containing all the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare orthographic data for analysis."
    )
    parser.add_argument(
        "input", type=str, help="The source XML TEI file for processing."
    )
    parser.add_argument(
        "-d",
        "--decompose",
        choices=["none", "pca", "tsne"],
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
        default="output",
        help="The prefix for all output files, including plots and CSV (default: 'output')",
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
        choices=VERBOSITY_LEVEL.keys(),
        default=1,
        help="Verbosity level: 0=WARNING, 1=INFO, 2=DEBUG, 3=ERROR (default: 1)",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function that parses arguments, processes data, and executes clustering.

    Returns
    -------
    None
    """
    args = parse_arguments()
    configure_logging(args.verbosity)

    df = read_data(args.input)
    df = scale_and_decompose(df, args.scale, args.decompose)
    labels = perform_clustering(df, args.cluster, args.clusters)

    logging.info("Data points and their cluster labels:")
    for index, label in zip(df.index, labels):
        logging.info(f"{index}: Cluster {label}")

    output_prefix = os.path.splitext(args.output)[0]  # Remove extension if any
    generate_plots(df, labels, output_prefix)

    save_results(df, labels, output_prefix)


if __name__ == "__main__":
    main()
