#!/usr/bin/env python3

"""
Converts a CSV with orthographic annotations to a data table for analysis.
"""

# Import Python standard libraries
from collections import defaultdict
from pathlib import Path
import argparse
import itertools
import logging
import csv

# Import 3rd-party libraries
import pandas as pd


def read_csv(filename: str) -> dict:
    """
    Build a data structure based on the readings from a CSV file.
    """
    data_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    with open(filename, "r", encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            token_index, name_attr, varSeq, reading, witness, most_common_reading = row
            data_structure[most_common_reading][reading][witness] += 1

    return convert_to_dict(data_structure)


def convert_to_dict(input_dict):
    """
    Recursively converts nested defaultdicts to standard dictionaries.
    """
    if isinstance(input_dict, (defaultdict, dict)):
        input_dict = {key: convert_to_dict(value) for key, value in input_dict.items()}
    return input_dict


def process_raw(data: dict, smoothing: str, filter: bool) -> pd.DataFrame:
    """
    Process raw data with observations.
    """

    logging.debug(f"Processing data...")

    def pass_filter(ms_label):
        if "-" in ms_label:
            return False
        if "(" in ms_label:
            return False

        return True

    # Collect all key/key pairs; also collect the total number of observations per proto-form
    # for each manuscript, so that we can later compute the ratios
    flatten = {}
    form_obs = defaultdict(int)
    for key, value in data.items():
        for subkey, subvalue in value.items():
            if filter:
                flatten[f"{key}___{subkey}"] = {
                    ms: reading for ms, reading in subvalue.items() if pass_filter(ms)
                }
            else:
                flatten[f"{key}___{subkey}"] = subvalue

            for ms, count in subvalue.items():
                if pass_filter(ms):
                    form_obs[key, ms] += count

    # Collect all manuscripts (it would be faster to do it in the loop above,
    # but this is clearer)
    mss = sorted(
        set(
            itertools.chain.from_iterable(
                [list(value.keys()) for value in flatten.values()]
            )
        )
    )

    # Build the Pandas dataframe (again, it would be faster in the loop above, but this
    # makes future manipulations easier)
    readings = sorted(list(flatten))
    if smoothing == "mle":
        vectors = []
        for ms in mss:
            vector = []
            for reading in readings:
                total = form_obs.get((reading.split("___")[0], ms))
                if not total:
                    vector.append(0)
                else:
                    vector.append(flatten[reading].get(ms, 0) / total)

            vectors.append(vector)
    elif smoothing == "none":
        vectors = [[flatten[reading].get(ms, 0) for reading in readings] for ms in mss]

    # Build dataframe from vectors
    df = pd.DataFrame(vectors, mss, readings)

    return df


def write_tabular(data: pd.DataFrame, output_file: str, overwrite: bool):
    """
    Write results dataframe to disk.
    """

    # Write data to disk, if possible/allowed
    if Path(output_file).is_file():
        if overwrite:
            logging.debug(f"Output file `{output_file}` already exists, overwriting...")
            write = True
        else:
            logging.debug(
                f"Output file `{output_file}` already exists, skipping (use `--overwrite`?)..."
            )
            write = False
    else:
        logging.debug(f"Writing to output file `{output_file}`...")
        write = True

    if write:
        data.to_csv(output_file, sep="\t", encoding="utf-8", float_format="%.4f")
        logging.debug(
            "Wrote file (note that in some systems you might need to wait some seconds for the flushing)."
        )


def parse_arguments() -> dict:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare orthographic data for analysis."
    )
    parser.add_argument("input", type=str, help="The source CSV file for processing.")
    parser.add_argument(
        "output",
        type=str,
        help="The output file (defaults to a name similar to input, without overwriting).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite an existing output file.",
    )
    parser.add_argument(
        "-s",
        "--smoothing",
        type=str,
        choices=["mle", "none"],
        default="mle",
        help="What kind of smoothing to apply (default: mle).",
    )
    parser.add_argument(
        "-f",
        "--filter",
        action="store_true",
        help="Filter manuscript keeping only the main transcriptions.",
    )
    arguments = vars(parser.parse_args())

    return arguments


def main(arguments: dict):
    """
    Script entry point
    """
    raw = read_csv(arguments["input"])
    data = process_raw(raw, arguments["smoothing"], arguments["filter"])
    write_tabular(data, arguments["output"], arguments["overwrite"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_arguments()
    main(args)
