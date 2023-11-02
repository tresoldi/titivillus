import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import seaborn as sns
import logging

FIGSIZE = (10, 10)

from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture


def draw_ellipse(position, covariance, ax, color, nsig=2, **kwargs):
    """
    Draw an ellipse with a given position and covariance.
    """
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * nsig * np.sqrt(s)
    else:
        # If the covariance of the GMM is diagonal, s is directly the variance
        angle = 0
        width, height = 2 * nsig * np.sqrt(covariance)

    # Draw the Ellipse
    ax.add_patch(Ellipse(position, width, height, angle=angle, color=color, **kwargs))


def plot_2d_clusters(
    principal: pd.DataFrame,
    labels: np.ndarray,
    annotate: bool = True,
    ellipses: bool = False,
) -> None:
    unique_labels = np.unique(labels)
    colormap = ListedColormap(sns.color_palette("hsv", len(unique_labels)).as_hex())
    norm = Normalize(vmin=labels.min(), vmax=labels.max())

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter plot with annotations
    scatter = ax.scatter(
        principal.iloc[:, 0],
        principal.iloc[:, 1],
        c=labels,
        cmap=colormap,
        norm=norm,
        alpha=0.7,
    )

    # Add a legend for clusters
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    if annotate:
        for idx, (x, y) in enumerate(zip(principal.iloc[:, 0], principal.iloc[:, 1])):
            ax.annotate(
                str(labels[idx]),
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    # Drawing ellipses if needed
    if ellipses:
        gmm = GaussianMixture(
            n_components=len(unique_labels), covariance_type="full", random_state=5
        ).fit(principal.iloc[:, :2])
        for i, (pos, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
            draw_ellipse(
                pos, covar, ax=ax, alpha=gmm.weights_[i] * 0.5, color=colormap.colors[i]
            )

    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    plt.title("2D Scatter Plot of Clusters")
    plt.show()


def plot_3d_clusters(principal: pd.DataFrame, labels: np.ndarray) -> None:
    if principal.shape[1] < 3:
        logging.error("3D plot requested but not enough dimensions in data.")
        raise ValueError("Data does not have enough dimensions for 3D plotting.")

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        principal.iloc[:, 0],
        principal.iloc[:, 1],
        principal.iloc[:, 2],
        c=labels,
        cmap="rainbow",
    )
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_zlabel("Principal Component 3", fontsize=15)
    plt.title("3D Scatter Plot of Clusters")
    return fig


def plot_heatmap(principal: pd.DataFrame) -> None:
    similarity_matrix = np.corrcoef(principal.transpose())
    sns.heatmap(similarity_matrix, cmap="coolwarm")
    plt.title("Heatmap of Similarity Matrix")
    return plt.gcf()


def plot_clusters(
    principal: pd.DataFrame,
    labels: np.ndarray,
    plot_type: str = "2d",
    ellipses: bool = False,
    save_path: str = None,
) -> None:
    """
    Plot the data with labels and save to a file if 'save_path' is provided.
    Supports 2D, 3D, and heatmap plots based on 'plot_type'.
    """
    try:
        if plot_type == "3d":
            fig = plot_3d_clusters(principal, labels)
        elif plot_type == "heatmap":
            fig = plot_heatmap(principal)
        else:  # Default to 2D scatter plot
            fig = plot_2d_clusters(principal, labels, ellipses=True)

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}.")
        plt.close(fig)  # Close the figure to free memory if not showing it

    except Exception as e:
        logging.error(f"Error in plotting: {e}")
        raise

    # Show the plot only if not saving to ensure the display is not redundant
    if not save_path:
        plt.show()
