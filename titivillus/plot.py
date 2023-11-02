import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

FIGSIZE = (10, 10)


def plot_clusters(
    principal: pd.DataFrame,
    labels: np.ndarray,
    plot_type: str = "2d",
    save_path: str = None,
) -> None:
    """
    Plot the data with labels and save to a file if 'save_path' is provided.
    Supports 2D, 3D, and heatmap plots based on 'plot_type'.

    Parameters
    ----------
    principal : pd.DataFrame
        The principal components of the data.
    labels : np.ndarray
        The labels for each data point indicating which cluster they belong to.
    plot_type : str, optional
        The type of plot to generate. '2d' for 2D scatter plot, '3d' for 3D scatter plot,
        and 'heatmap' for heatmap. The default is '2d'.
    save_path : str, optional
        The path where the plot should be saved. If not specified, the plot will be shown.
    """
    try:
        fig = None
        # For 3D plotting
        if plot_type == "3d":
            fig = plt.figure(figsize=FIGSIZE)
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
            plt.title("3D Scatter Plot of Clusters")

        # For heatmap plotting
        elif plot_type == "heatmap":
            similarity_matrix = np.corrcoef(principal.transpose())
            sns.heatmap(similarity_matrix, cmap="coolwarm")
            plt.title("Heatmap of Similarity Matrix")
            fig = plt.gcf()

        # Default 2D plotting
        else:
            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.set_xlabel("Principal Component 1", fontsize=15)
            ax.set_ylabel("Principal Component 2", fontsize=15)
            scatter = ax.scatter(
                principal.iloc[:, 0], principal.iloc[:, 1], c=labels, cmap="rainbow"
            )
            legend = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend)
            plt.title("2D Scatter Plot of Clusters")

        # Save the plot if a save_path is provided
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()
    except Exception as e:
        logging.error(f"Error in plotting: {e}")
        raise
