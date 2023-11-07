#!/usr/bin/env python3
"""
json2csv.py

This script processes JSON data to generate a CSV file containing reading
variations from textual witnesses as recorded in the JSON file.
"""

# Standard library imports
import json
import argparse
import logging
from collections import defaultdict

# Third-party library imports
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_json_to_csv(input_json):
    """
    Parse the JSON data and generate a DataFrame suitable for analysis.

    Parameters
    ----------
    input_json : str
        The file path of the input JSON file.

    Returns
    -------
    pd.DataFrame
        The processed data in a DataFrame.
    """
    # Load JSON data
    with open(input_json, "r", encoding="utf-8") as file_handle:
        json_data = json.load(file_handle)

    # Initialize a dictionary to hold the combined data
    combined_data = defaultdict(lambda: defaultdict(int))

    # Iterate over the JSON data and populate the combined data structure
    for normalized_form, attested_readings in json_data.items():
        for attested_reading, manuscripts in attested_readings.items():
            for manuscript_label, count in manuscripts.items():
                # Use a tuple of normalized form and attested reading as the key
                key = f"{normalized_form}___{attested_reading}"
                combined_data[manuscript_label][key] += count

    # Create a DataFrame from the combined data
    df = pd.DataFrame.from_dict(combined_data, orient="index").fillna(0)

    # Sort the DataFrame by index (manuscript labels)
    df.sort_index(inplace=True)

    return df


def write_dataframe_to_csv(dataframe, output_path, sep="\t"):
    """
    Write the DataFrame to a CSV file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data to be written to CSV.
    output_path : str
        The path to the output CSV file.
    sep : str
        The separator to use in the CSV file.
    """
    dataframe.to_csv(output_path, sep=sep, encoding="utf-8", float_format="%.0f")
    logging.info(f"Data written to {output_path}")


def parse_command_line_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process JSON data to generate a CSV of readings."
    )
    parser.add_argument("input_json", type=str, help="Path to the input JSON file.")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help="Separator to use in the output CSV file.",
    )
    return parser.parse_args()


def main():
    """
    The main function to execute the script functionality.
    """
    args = parse_command_line_arguments()

    logging.info("Processing JSON data...")
    dataframe = parse_json_to_csv(args.input_json)

    logging.info("Writing output CSV...")
    write_dataframe_to_csv(dataframe, args.output_csv, sep=args.sep)


if __name__ == "__main__":
    main()
