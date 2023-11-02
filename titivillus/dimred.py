import pandas as pd
import logging
from typing import Optional, Union, Callable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Factory pattern setup
decomposition_methods = {}


def register_decomposition(name: str) -> Callable:
    def inner(func: Callable) -> Callable:
        decomposition_methods[name] = func
        return func

    return inner


@register_decomposition("pca")
def pca_decomposition(
    data: pd.DataFrame, n_components: Optional[Union[int, str]] = None
) -> pd.DataFrame:
    """
    Perform PCA decomposition on the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to perform decomposition on.
    n_components : Optional[Union[int, str]]
        The number of components to keep. If not set, min(number of features, 10) is used.

    Returns
    -------
    pd.DataFrame
        The transformed data after applying PCA.
    """
    if n_components is None:
        n_components = min(len(data.columns), 10)

    pca_decomp = PCA(n_components=n_components)
    components = pca_decomp.fit_transform(data)
    colnames = [f"pc{idx + 1}" for idx in range(components.shape[1])]
    principal_df = pd.DataFrame(data=components, columns=colnames, index=data.index)
    logging.info(
        f"PCA: Explained variance ratio: {pca_decomp.explained_variance_ratio_}"
    )
    return principal_df


@register_decomposition("tsne")
def tsne_decomposition(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Perform t-SNE decomposition on the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data to perform decomposition on.
    n_components : int
        The dimension of the embedded space.

    Returns
    -------
    pd.DataFrame
        The transformed data after applying t-SNE.
    """
    tsne = TSNE(n_components=n_components)
    components = tsne.fit_transform(data)
    colnames = [f"tsne{idx + 1}" for idx in range(components.shape[1])]
    tsne_df = pd.DataFrame(data=components, columns=colnames, index=data.index)
    logging.info("t-SNE decomposition completed.")
    return tsne_df


# @register_decomposition('umap')
# def umap_decomposition(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
#     """
#     Perform UMAP decomposition on the data.
#
#     Parameters
#     ----------
#     data : pd.DataFrame
#         The data to perform decomposition on.
#     n_components : int
#         The number of components to use for UMAP.
#
#     Returns
#     -------
#     pd.DataFrame
#         The transformed data after applying UMAP.
#     """
#     umap_decomp = UMAP(n_components=n_components)
#     components = umap_decomp.fit_transform(data)
#     colnames = [f"umap{idx + 1}" for idx in range(components.shape[1])]
#     umap_df = pd.DataFrame(data=components, columns=colnames, index=data.index)
#     logging.info("UMAP decomposition completed.")
#     return umap_df


def decompose_data(data: pd.DataFrame, method: str, **kwargs) -> pd.DataFrame:
    """
    Decompose data using the specified method.

    Parameters
    ----------
    data : pd.DataFrame
        The data to decompose.
    method : str
        The decomposition method to use ('pca', 'tsne', 'umap').

    Returns
    -------
    pd.DataFrame
        The decomposed data.
    """
    if method not in decomposition_methods:
        raise ValueError(f"Decomposition method '{method}' not recognized.")

    try:
        decompose_func = decomposition_methods[method]
        return decompose_func(data, **kwargs)
    except Exception as e:
        logging.error(f"Error in '{method}' decomposition: {e}")
        raise
