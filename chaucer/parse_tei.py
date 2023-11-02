import xml.etree.ElementTree as ET
import csv
from collections import defaultdict
import argparse


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
    parser.add_argument("output_csv", type=str, help="Path to the output CSV file.")

    return parser.parse_args()


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


def main():
    """
    Main function to process XML data and write results to a CSV.
    """
    args = parse_args()

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

    # Open a CSV file for writing
    with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row
        csv_writer.writerow(
            [
                "Token Index",
                "Name",
                "Reading Number",
                "Reading",
                "Witnesses",
                "Most Common Reading",
            ]
        )

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
                    csv_writer.writerow(
                        [
                            token_index,
                            name_attr,
                            varSeq,
                            reading,
                            witness,
                            most_common_reading,
                        ]
                    )


if __name__ == "__main__":
    main()
