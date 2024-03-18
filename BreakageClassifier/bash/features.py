import argparse
from pathlib import Path
import sys
from typing import List, Optional
from BreakageClassifier.code.run import pipeline


def main(program: str, args: List[str]):
    parser = argparse.ArgumentParser(
        prog=program, description="Run a classification pipeline."
    )

    parser.add_argument(
        "--crawl-dir",
        type=Path,
        help="Input crawl directory.",
        default=Path("crawl-data"),
    )

    # parser.add_argument(
    #     "--input-db",
    #     type=Path,
    #     help="Input SQLite database.",
    #     default=Path("crawl-data.sqlite"),
    # )

    # parser.add_argument(
    #     "--input-exp",
    #     type=Path,
    #     help="Input Experiment log file .csv",
    #     default=Path("experiments.csv"),
    # )

    # parser.add_argument("--ldb", type=str, help="Input LDB.", default="content.ldb")

    parser.add_argument("--features", type=Path, help="Features.", default=None)

    # parser.add_argument(
    #     "--filters", type=Path, help="Filters directory.", default=Path("filterlists")
    # )

    parser.add_argument(
        "--out", type=Path, help="Directory to output the results.", default=Path("out")
    )

    def list_of_strings(arg):
        return arg.split(",")

    parser.add_argument(
        "--issues",
        type=list_of_strings,
        help="Issues to classify.",
        default=None,
    )

    parser.add_argument(
        "--edits",
        type=Path,
        help="Optional directory for extracted edit trees to export features",
        default=None,
    )

    ns = parser.parse_args(args)

    if ns.features:
        pipeline(
            ns.crawl_dir,
            # ns.input_db,
            # ns.input_exp,
            # ns.ldb,
            # ns.filters,
            ns.out,
            ns.features,
            # False,
            ns.edits,
            ns.issues,
        )

    else:
        pipeline(
            ns.crawl_dir,
            # ns.input_db,
            # ns.input_exp,
            # ns.ldb,
            # ns.filters,
            ns.out,
            # overwrite=False,
            edits_dir=ns.edits,
            issues=ns.issues,
        )


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
