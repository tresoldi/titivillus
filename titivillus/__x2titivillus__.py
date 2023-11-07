#!/usr/bin/env python3
"""
convert_to_csv.py

This script processes XML and JSON data to generate CSV files containing reading
variations from textual witnesses as recorded in the TEI XML or JSON file.
"""

# Standard library imports
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Tuple
import argparse
import itertools
import json
import logging
import xml.etree.ElementTree as ET

# Third-party library imports
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def manuscript_passes_filter(ms_label: str) -> bool:
    """
    Determine if a manuscript label passes the filter criteria.

    Parameters
    ----------
    ms_label : str
        The manuscript label to be checked.

    Returns
    -------
    bool
        True if the manuscript label passes the filter, False otherwise.
    """
    return not any(char in ms_label for char in "-()")


def get_most_common_reading(
    reading_frequencies: DefaultDict[str, DefaultDict[str, int]], name: str
) -> str:
    """
    Get the most common reading for a given name based on reading frequencies.

    Parameters
    ----------
    reading_frequencies : defaultdict(int)
        A dictionary storing frequency count of readings for each name.
    name : str
        The name for which the most common reading is determined.

    Returns
    -------
    str
        The most common reading for the given name.
    """
    return max(reading_frequencies[name], key=reading_frequencies[name].get)


def convert_nested_defaultdict_to_dict(nested_defaultdict: DefaultDict) -> Dict:
    """
    Recursively converts a nested defaultdict to a standard dictionary.

    Parameters
    ----------
    nested_defaultdict : defaultdict
        The nested defaultdict to convert.

    Returns
    -------
    dict
        The converted standard dictionary.
    """
    if isinstance(nested_defaultdict, (defaultdict, dict)):
        return {
            key: convert_nested_defaultdict_to_dict(value)
            for key, value in nested_defaultdict.items()
        }
    return nested_defaultdict


def parse_xml_to_csv(
    input_xml: str,
) -> DefaultDict[str, DefaultDict[str, DefaultDict[str, int]]]:
    """
    Parse the XML data and generate intermediate data suitable for analysis.

    Parameters
    ----------
    input_xml : str
        The file path of the input XML file.

    Returns
    -------
    defaultdict
        The intermediate data structure containing reading frequencies.
    """
    # Load and parse the XML data
    with open(input_xml, "r", encoding="utf-8") as file_handle:
        xml_content = file_handle.read().replace("\f", "")
    root = ET.fromstring(xml_content)

    # Initialize a dictionary to store frequency counts of readings
    reading_frequencies = defaultdict(lambda: defaultdict(int))
    namespace = {"ns": "http://www.tei-c.org/ns/1.0"}

    # Count frequencies of each reading for names
    for index, app in enumerate(root.findall(".//ns:app", namespace)):
        name = f"{app.attrib['n']}-{index}"
        for reading_elem in app.findall("ns:rdg", namespace):
            reading_text = reading_elem.text
            reading_frequencies[name][reading_text] += 1

    # Extract data for CSV
    csv_data = []
    for index, app in enumerate(root.findall(".//ns:app", namespace)):
        name_attribute = app.attrib["n"]
        token_index = str(index)
        common_reading = get_most_common_reading(
            reading_frequencies, f"{name_attribute}-{token_index}"
        )

        for reading_elem in app.findall("ns:rdg", namespace):
            reading_text = reading_elem.text or ""
            reading_number = reading_elem.get("varSeq")
            witnesses = [
                idno.text
                for wit_elem in reading_elem.findall("ns:wit", namespace)
                for idno in wit_elem.findall("ns:idno", namespace)
            ]
            for witness in witnesses:
                csv_data.append(
                    {
                        "Token Index": token_index,
                        "Name": name_attribute,
                        "Reading Number": reading_number,
                        "Reading": reading_text,
                        "Witnesses": witness,
                        "Most Common Reading": common_reading,
                    }
                )

    # Convert the list of dictionaries to a nested default dictionary
    nested_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in csv_data:
        nested_dict[row["Most Common Reading"]][row["Reading"]][row["Witnesses"]] += 1

    return nested_dict


def process_intermediate_tei_data(
    raw_data: Dict, filter_witnesses: bool
) -> pd.DataFrame:
    """
    Process the intermediate raw data into a pandas DataFrame.

    Parameters
    ----------
    raw_data : dict
        The intermediate raw data.
    filter_witnesses : bool
        Whether to filter out certain witnesses.

    Returns
    -------
    pd.DataFrame
        The processed data in a DataFrame.
    """

    flatten_data = {}
    form_observations = defaultdict(int)
    for key, readings_dict in raw_data.items():
        for subkey, witnesses_count in readings_dict.items():
            valid_witnesses = (
                {
                    ms: count
                    for ms, count in witnesses_count.items()
                    if manuscript_passes_filter(ms)
                }
                if filter_witnesses
                else witnesses_count
            )
            flatten_data_key = f"{key}___{subkey}"
            flatten_data[flatten_data_key] = valid_witnesses

            for manuscript, count in valid_witnesses.items():
                form_observations[(key, manuscript)] += count

    manuscripts = sorted(
        set(itertools.chain.from_iterable(d.keys() for d in flatten_data.values()))
    )
    readings = sorted(flatten_data)

    vectors = [
        [flatten_data[reading].get(ms, 0) for reading in readings] for ms in manuscripts
    ]

    dataframe = pd.DataFrame(vectors, index=manuscripts, columns=readings)
    return dataframe


def parse_json_to_csv(input_json: str, filter_witnesses: bool) -> pd.DataFrame:
    """
    Parse the JSON data and generate a DataFrame suitable for analysis.

    Parameters
    ----------
    input_json : str
        The file path of the input JSON file.
    filter_witnesses : bool
        Whether to filter out certain witnesses.

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
                if not filter_witnesses or manuscript_passes_filter(manuscript_label):
                    # Use a tuple of normalized form and attested reading as the key
                    key = f"{normalized_form}___{attested_reading}"
                    combined_data[manuscript_label][key] += count

    # Create a DataFrame from the combined data
    df = pd.DataFrame.from_dict(combined_data, orient="index").fillna(0)

    # Sort the DataFrame by index (manuscript labels)
    df.sort_index(inplace=True)

    return df


def write_tabular_output(
    dataframe: pd.DataFrame, output_path: str, sep: str = "\t"
) -> None:
    """
    Write the given DataFrame to a tabular file at the specified path.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be written to the tabular file.
    output_path : str
        The file system path where the tabular file will be saved.
    sep : str, optional
        The string character (default is a tab character) used to separate
        columns in the tabular file.

    Returns
    -------
    None

    Notes
    -----
    The function will write the DataFrame to a tabular file using UTF-8
    encoding and will format floating-point numbers to not have any decimal
    places. It logs an info message upon successful writing of the file.
    """
    dataframe.to_csv(output_path, sep=sep, encoding="utf-8", float_format="%.0f")
    logging.info(f"Data written to {output_path}")


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Parse and return the command line arguments provided by the user.

    This function uses argparse to define and parse command line arguments for
    the script, allowing the user to specify the input file, output file, and
    various options for processing.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command line arguments as attributes.
        The attributes that are expected to be available are:
        - input_file (str): Path to the input XML or JSON file.
        - output_csv (str): Path to the output tabular file.
        - sep (str): Separator to use in the output tabular file; defaults to a
          tab character.
        - overwrite (bool): Flag indicating whether to overwrite the output
          file if it exists; defaults to False.
        - type (str): Type of the input file, either 'xml' or 'json'; optional,
          inferred from file extension if not provided.
        - filter_witnesses (bool): Flag indicating whether to filter out
          certain witnesses; defaults to False.

    Notes
    -----
    The function sets up the following command line arguments:
    - Positional arguments:
      - input_file: The path to the input file, which can be an XML or JSON file.
      - output_csv: The path to the output tabular file.
    - Optional arguments:
      - --sep: The column separator character for the output file (default: tab).
      - --overwrite: A flag to specify whether to overwrite the output file if
        it exists.
      - --type: The type of the input file; if not provided, it is inferred
        from the file extension.
      - --filter-witnesses: A flag to specify whether to filter out certain
        witnesses during processing.

    The function will exit the script and print a help message to the command
    line if the arguments are not used correctly.
    """
    parser = argparse.ArgumentParser(
        description="Process XML or JSON data to generate a CSV of readings."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input file (XML or JSON)."
    )
    parser.add_argument("output_csv", type=str, help="Path to the output file.")
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help="Separator to use in the output tabular file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["xml", "json"],
        help="The type of the input file (optional).",
    )
    parser.add_argument(
        "--filter-witnesses", action="store_true", help="Filter out certain witnesses."
    )
    return parser.parse_args()


def main() -> None:
    """
    Script entry point.
    """
    args = parse_command_line_arguments()

    # Check if the output file exists before proceeding
    output_file = Path(args.output_csv)
    if output_file.is_file() and not args.overwrite:
        logging.info(
            f"Output file `{args.output_csv}` already exists. Use `--overwrite` to enable overwriting."
        )
        return

    # Infer the file type from the extension if not provided
    if not args.type:
        if args.input_file.endswith(".xml"):
            args.type = "xml"
        elif args.input_file.endswith(".json"):
            args.type = "json"
        else:
            raise ValueError(
                "Could not infer file type from extension. Please provide the --type argument."
            )

    # Based on the file type, process the input file
    if args.type == "xml":
        logging.info("Processing XML data...")
        intermediate_data = parse_xml_to_csv(args.input_file)
        converted_data = convert_nested_defaultdict_to_dict(intermediate_data)
        dataframe = process_intermediate_tei_data(converted_data, args.filter_witnesses)
    elif args.type == "json":
        logging.info("Processing JSON data...")
        dataframe = parse_json_to_csv(args.input_file, args.filter_witnesses)

    # After processing, write the output tabular file
    logging.info("Writing output tabular file...")
    write_tabular_output(dataframe, args.output_csv, sep=args.sep)


if __name__ == "__main__":
    main()
