#!/usr/bin/env python3

"""
Converts a JSON with orthographic annotations to a data table for analysis.
"""

# Import Python standard libraries
from collections import defaultdict
from pathlib import Path
import argparse
import itertools
import json
import logging

# Import 3rd-party libraries
import pandas as pd


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


def read_json(input_file: str) -> dict:
    """
    Read and return raw json data.
    """

    logging.debug(f"Reading source file `{input_file}`...")

    with open(input_file) as h:
        data = json.load(h)

    return data


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

    :return: The command-line arguments as a dictionary.
    """

    # Obtain arguments as a dictionary
    parser = argparse.ArgumentParser(
        description="Prepare orthographic data for analysis."
    )
    parser.add_argument("input", type=str, help="The source JSON file for processing.")
    parser.add_argument(
        "-o",
        "--output",
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

    # Set defaults not handled by argparse
    if not arguments["output"]:
        p = Path(arguments["input"])
        arguments["output"] = str(p.parent / str(p.stem + ".tsv"))

    return arguments


def main(arguments: dict):
    """
    Script entry point
    """

    # Read raw JSON data, process it, and write back
    raw = read_json(arguments["input"])
    data = process_raw(raw, arguments["smoothing"], arguments["filter"])
    write_tabular(data, arguments["output"], arguments["overwrite"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_arguments()
    main(args)
