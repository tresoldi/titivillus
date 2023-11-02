import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


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
