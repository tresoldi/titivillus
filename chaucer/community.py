#!/usr/bin/env python3

"""
Performs community detection on preprocessed orthographic tabular data.
"""

# Import Python standard libraries
from typing import *
import argparse
import logging
import random

# Import 3rd-party libraries
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def cluster_affinity(df:pd.DataFrame):
    clustering = AffinityPropagation(random_state=5).fit(df)

    return clustering.labels_

def cluster_random(df:pd.DataFrame):
    rows = len(df.index)
    clusters = random.randint(2, int(rows/2))

    return [random.randint(0, clusters) for _ in range(rows)]


def pca(data: pd.DataFrame, n_components: Union[int, str]):
    """
    Perform standard PCA decomposition.
    """

    # Run PCA, extract components, and build new dataframe; the name of the columns must be computed
    # after the components are collected, as the `mle` argument value can lead to size
    # that cannot be predicted beforehand
    pca_decomp = PCA(n_components=n_components)
    components = pca_decomp.fit_transform(data)
    colnames = [f"pc{idx+1}" for idx in range(len(components[0]))]
    principal = pd.DataFrame(data=components, columns=colnames, index=data.index.values)

    print(pca_decomp.explained_variance_ratio_)

    return principal


def plot(principal, labels):
    mss = principal.index.values

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    #ax.set_title("2 component PCA", fontsize=20)
    plt.scatter(principal.pc1, principal.pc2)
    ax.grid()

    matrix = principal.to_numpy()
    colors = plt.cm.rainbow(np.linspace(0, 1, max(labels)+1))
    for i in range(len(matrix)):
        x = matrix[i][0]
        y = matrix[i][1]
        plt.plot(x, y, "bo",c=colors[labels[i]])
        plt.text(x * (1 + 0.03), y * (1 + 0.03), mss[i], fontsize=8)

    plt.show()


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a dataframe prepared with `convert.py`.
    """

    df = pd.read_csv(filename, delimiter="\t", encoding="utf-8", index_col=0)

    return df


def parse_arguments() -> dict:
    """
    Parse command line arguments.

    :return: The command-line arguments as a dictionary.
    """

    # Obtain arguments as a dictionary
    parser = argparse.ArgumentParser(
        description="Prepare orthographic data for analysis."
    )
    parser.add_argument("input", type=str, help="The source JSON file for processing.")
    parser.add_argument(
        "-s",
        "--scale",
        type=str,
        choices=["none", "standard_nomean", "standard_mean"],
        default="none",
        help="Whether to perform preprocessing scaling and of which kind (default: none)",
    )
    parser.add_argument(
        "-d",
        "--decompose",
        type=str,
        choices=["none", "pca"],
        default="pca",
        help="Whether to perform decomposition and of which kind (default: pca)",
    )
    # NOTE: the random method is only for testing/developing purposes!
    parser.add_argument(
        "-c", "--cluster",
        type=str,
        choices=["affinity", "random"],
        default="affinity",
        help="Whether to perform clustering and of which kind (default: affinity)",
    )
    arguments = vars(parser.parse_args())

    return arguments


def main(arguments: dict):
    """
    Script entry point
    """

    # Read tabular data as a pandas dataframe
    df = read_dataframe(arguments["input"])

    # Perform scaling if requested (as the matrix is very sparse, we usually don't want this)
    if arguments["scale"] == "standard_mean":
        df = StandardScaler().fit_transform(df)
    elif arguments["scale"] == "standard_nomean":
        df = StandardScaler(with_mean=False).fit_transform(df)

    # Run decomposition if requested
    if arguments["decompose"] == "pca":
        df = pca(df, 2)

    # Run the clustering
    if arguments["cluster"]=="affinity":
        labels = cluster_affinity(df)
    elif arguments["cluster"]=="random":
        labels = cluster_random(df)

    # Plot results
    print(list(zip(df.index.values, labels)))

    plot(df, labels)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    main(args)
