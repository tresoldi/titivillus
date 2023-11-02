# Import standard libraries
from typing import Callable, Dict
import logging

# Import third-party libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import numpy as np
import pandas as pd


# A factory of scaling functions
ScalerFactory: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {}


def register_scaler(name: str) -> Callable:
    """
    A decorator to register new scaler functions to the factory.

    Parameters
    ----------
    name : str
        The name of the scaling method.

    Returns
    -------
    Callable
        A decorator that registers the scaling function.
    """

    def decorator(func: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable:
        ScalerFactory[name] = func
        return func

    return decorator


@register_scaler("standard")
def scale_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale data using the standard scaling method (z-score normalization).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale.

    Returns
    -------
    pd.DataFrame
        The scaled dataframe.
    """
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


@register_scaler("standard_nomean")
def scale_standard_nomean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale data using standard scaling without centering.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale.

    Returns
    -------
    pd.DataFrame
        The scaled dataframe.
    """
    scaler = StandardScaler(with_mean=False)
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


@register_scaler("minmax")
def scale_minmax(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale data using Min-Max scaling.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale.

    Returns
    -------
    pd.DataFrame
        The scaled dataframe.
    """
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


@register_scaler("robust")
def scale_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale data using Robust scaling, which is less sensitive to outliers.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale.

    Returns
    -------
    pd.DataFrame
        The scaled dataframe.
    """
    scaler = RobustScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


@register_scaler("l2_norm")
def scale_l2_norm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale data using L2 normalization, which scales individual samples to have unit norm.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale.

    Returns
    -------
    pd.DataFrame
        The scaled dataframe.
    """
    scaler = Normalizer(norm="l2")
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


def scale_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Scale the data using the specified method from the scaling factory.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to scale.
    method : str
        The name of the scaling method to apply.

    Returns
    -------
    pd.DataFrame
        The scaled dataframe.

    Raises
    ------
    ValueError
        If the scaling method is not recognized.
    """
    if method not in ScalerFactory:
        logging.error(f"Scaling method '{method}' not recognized.")
        raise ValueError(f"Scaling method '{method}' not recognized.")

    try:
        # Use the factory to get the scaler function
        scale_func = ScalerFactory[method]
        return scale_func(df)
    except Exception as e:
        logging.error(f"Error scaling data: {e}")
        raise


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Read a dataframe from a tab-delimited file.

    Parameters
    ----------
    filename : str
        The path to the tab-delimited file.

    Returns
    -------
    pd.DataFrame
        The dataframe read from the file.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    Exception
        If there is an error reading the dataframe.
    """
    try:
        df = pd.read_csv(filename, delimiter="\t", encoding="utf-8", index_col=0)
        return df
    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
        raise
    except Exception as e:
        logging.error(f"Error reading the dataframe: {e}")
        raise


def save_to_csv(data: pd.DataFrame, labels: np.ndarray, output_file: str) -> None:
    """
    Save the data along with its cluster labels to a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe to save.
    labels : np.ndarray
        The cluster labels to save with the dataframe.
    output_file : str
        The path to the output CSV file.

    Raises
    ------
    Exception
        If there is an error saving to CSV.
    """
    try:
        # Add the labels to the dataframe
        output_df = data.assign(Cluster=labels)
        output_df.to_csv(output_file, index=True)
        logging.info(f"Cluster labels saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        raise
