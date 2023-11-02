import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


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
