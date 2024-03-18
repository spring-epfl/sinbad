import argparse
import csv
from pathlib import Path
import sys
from typing import List
import pandas as pd
from storage_dump.storage import DataframeCSVStorageController


def label(
    in_path: Path,
    out_path: Path,
    start: int = None,
    end: int = None,
):
    df = pd.read_csv(
        str(in_path.resolve()),
        usecols=["id", "raw_post", "issue_url", "should_include"],
    )

    df = df[df["should_include"]]
    rows = df.shape[0]

    if start and end and (start > rows or end < 0):
        return

    if not start or start < 0:
        start = 0

    if not end or end > rows:
        end = rows

    print(
        "===================================================================================================="
    )

    with DataframeCSVStorageController(
        out_path.parent,
        [
            out_path.name,
        ],
    ) as storage:

        try:
            prev_labeled = storage.load(out_path.name).id.unique()
        except:
            prev_labeled = []

        for i in range(start, end):

            id = df.iloc[i]["id"]

            if id in prev_labeled:
                continue

            post = df.iloc[i]["raw_post"]
            url = df.iloc[i]["issue_url"]
            print("\n", i, "issue id = ", id)
            print(
                "\n----------------------------------------------------------------------------------------------------"
            )
            print("\n", url)
            print(
                "\n----------------------------------------------------------------------------------------------------"
            )
            print("\n", post)
            print(
                "\n----------------------------------------------------------------------------------------------------"
            )
            dynamic_label = input("\nDynamic? [1 if yes else nothing]: ")
            if dynamic_label == "1":
                dynamic_label = 1
            else:
                dynamic_label = 0

            print(
                "\n===================================================================================================="
            )

            print(f"Labeled as {'Dynamic' if dynamic_label == 1 else 'Static'}")

            storage.save(
                out_path.name,
                pd.DataFrame(
                    [
                        {"id": id, "label": "D" if dynamic_label == 1 else "S"},
                    ]
                ),
            )


def main(program: str, args: List[str]):
    """run the main crawling pipeline

    Args:
        program (str)
        args (List[str])
    """

    parser = argparse.ArgumentParser(
        prog=program, description="Run a crawling experiment"
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="input dataset path.",
        default=Path("../forums/ublock/data/ublock-data.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="output path for the results file.",
        default=Path("./datadir"),
    )

    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
    )

    ns = parser.parse_args(args)

    label(Path(ns.input), Path(ns.output), ns.start, ns.end)


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
