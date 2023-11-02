#!/usr/bin/env python3
"""
__tei2df__.py

This script processes XML data to generate a CSV file containing reading variations
from textual witnesses as recorded in the TEI XML file.
"""

# Standard library imports
from collections import defaultdict
import argparse
import itertools
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

# Third-party library imports
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_most_common_reading(reading_frequencies, name):
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


def parse_xml_to_csv(input_xml):
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


def convert_nested_defaultdict_to_dict(nested_defaultdict):
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


def process_intermediate_data(raw_data, smoothing, filter_witnesses):
    """
    Process the intermediate raw data into a pandas DataFrame.

    Parameters
    ----------
    raw_data : dict
        The intermediate raw data.
    smoothing : str
        The type of smoothing to apply to the data.
    filter_witnesses : bool
        Whether to filter out certain witnesses.

    Returns
    -------
    pd.DataFrame
        The processed data in a DataFrame.
    """

    def manuscript_passes_filter(ms_label):
        return not any(char in ms_label for char in "-()")

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

    if smoothing == "mle":
        vectors = [
            [
                flatten_data[reading].get(ms, 0)
                / form_observations.get((reading.split("___")[0], ms), 1)
                for reading in readings
            ]
            for ms in manuscripts
        ]
    elif smoothing == "none":
        vectors = [
            [flatten_data[reading].get(ms, 0) for reading in readings]
            for ms in manuscripts
        ]

    dataframe = pd.DataFrame(vectors, index=manuscripts, columns=readings)
    return dataframe


def write_dataframe_to_csv(dataframe, output_path, overwrite):
    """
    Write the DataFrame to a CSV file.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the data to be written to CSV.
    output_path : str
        The path to the output CSV file.
    overwrite : bool
        If true, will overwrite the existing file at the output path.
    """
    output_file = Path(output_path)
    if output_file.is_file() and not overwrite:
        logging.info(
            f"Output file `{output_path}` already exists. Use `--overwrite` to enable overwriting."
        )
        return

    dataframe.to_csv(output_path, sep="\t", encoding="utf-8", float_format="%.4f")
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
        description="Process XML data to generate a CSV of readings."
    )
    parser.add_argument("input_xml", type=str, help="Path to the input XML file.")
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "--smoothing",
        choices=["none", "mle"],
        default="none",
        help="The smoothing technique to use.",
    )
    parser.add_argument(
        "--filter-witnesses", action="store_true", help="Filter out certain witnesses."
    )
    return parser.parse_args()


def main():
    """
    The main function to execute the script functionality.
    """
    args = parse_command_line_arguments()

    logging.info("Processing XML data...")
    intermediate_data = parse_xml_to_csv(args.input_xml)
    converted_data = convert_nested_defaultdict_to_dict(intermediate_data)

    logging.info("Processing intermediate data...")
    dataframe = process_intermediate_data(
        converted_data, args.smoothing, args.filter_witnesses
    )

    logging.info("Writing output CSV...")
    write_dataframe_to_csv(dataframe, args.output_csv, args.overwrite)


if __name__ == "__main__":
    main()
