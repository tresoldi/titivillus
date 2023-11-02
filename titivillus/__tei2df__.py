import xml.etree.ElementTree as ET
from collections import defaultdict
import argparse
import pandas as pd
import logging
import itertools
from pathlib import Path


def get_most_common_reading(reading_frequencies: defaultdict, name: str) -> str:
    """
    Get the most common reading for a given name.

    Parameters
    ----------
    reading_frequencies : defaultdict
        A dictionary storing frequency count of readings for each name.
    name : str
        The name for which to find the most common reading.

    Returns
    -------
    str
        The most common reading for the given name.
    """
    return max(reading_frequencies[name], key=reading_frequencies[name].get)


def generate_intermediate_data(args):
    """
    Main function to process XML data and write results to a CSV.
    """

    # Load the XML data
    with open(args.input_xml, "r", encoding="utf-8") as h:
        xml_data = h.read().replace("\f", "")
    root = ET.fromstring(xml_data)

    # Dictionary to store frequency count of readings for each name
    reading_frequencies = defaultdict(lambda: defaultdict(int))
    namespace = {"ns": "http://www.tei-c.org/ns/1.0"}

    # First pass: Count the frequencies of each reading for every name
    for idx, app in enumerate(root.findall(".//ns:app", namespace)):
        name = app.attrib["n"] + "-" + str(idx)
        for rdg in app.findall("ns:rdg", namespace):
            reading = rdg.text
            reading_frequencies[name][reading] += 1

    # List to collect all rows as dictionaries
    csv_rows = []

    # Extract readings and witnesses
    for idx, app in enumerate(root.findall(".//ns:app", namespace)):
        name_attr = app.attrib["n"]
        token_index = str(idx)
        most_common_reading = get_most_common_reading(
            reading_frequencies, name_attr + "-" + token_index
        )

        for rdg in app.findall("ns:rdg", namespace):
            reading = rdg.text if rdg.text is not None else ""
            varSeq = rdg.get("varSeq")
            witnesses_list = [
                idno.text
                for wit in rdg.findall("ns:wit", namespace)
                for idno in wit.findall("ns:idno", namespace)
            ]
            for witness in witnesses_list:
                csv_rows.append(
                    {
                        "Token Index": token_index,
                        "Name": name_attr,
                        "Reading Number": varSeq,
                        "Reading": reading,
                        "Witnesses": witness,
                        "Most Common Reading": most_common_reading,
                    }
                )

    # build a data structure
    data_structure = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for row in csv_rows:
        token_index, name_attr, varSeq, reading, witness, most_common_reading = (
            row["Token Index"],
            row["Name"],
            row["Reading Number"],
            row["Reading"],
            row["Witnesses"],
            row["Most Common Reading"],
        )

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


def parse_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process XML data to generate a CSV of readings."
    )
    parser.add_argument("input_xml", type=str, help="Path to the input XML file.")
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

    return parser.parse_args()


def main():
    args = parse_args()
    intermediate_data = generate_intermediate_data(args)
    data = process_raw(intermediate_data, args.smoothing, args.filter)
    write_tabular(data, args.output, args.overwrite)


if __name__ == "__main__":
    main()
