import pandas as pd
import logging
from typing import Optional, Union
from sklearn.decomposition import PCA


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
