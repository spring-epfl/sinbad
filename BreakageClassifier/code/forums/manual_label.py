import argparse
from pathlib import Path
import sys
from time import sleep
from typing import List
import webbrowser
import pandas as pd
from storage_dump.storage import DataframeCSVStorageController
from tqdm import tqdm
import os


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class TestUrlUpdates:
    def __init__(self):
        self.urls_updates = {}

    def _update(self, row: pd.Series) -> pd.Series:

        if row.id in self.urls_updates:
            row.test_url = self.urls_updates[row.id]

        return row

    def commit(self, df: pd.DataFrame):

        return df.apply(self._update, axis=1)


def label(
    issues_fp: Path,
    out_fp: Path,
    attributes: List[str],
    n_start: int = None,
    count: int = None,
    filterlists: Path = None,
):

    issues = pd.read_csv(issues_fp)
    issues = issues[issues["should_include"]]

    if n_start and count:
        issues = issues.iloc[:, n_start : n_start + count]

    elif n_start:
        issues = issues.iloc[:, n_start:]

    elif count:
        issues = issues.iloc[:, :count]

    label_files = {a: f"manual-{a}.csv" for a in attributes}

    with DataframeCSVStorageController(
        out_fp, [label_files[a] for a in attributes] + ["manual-test-url.csv"]
    ) as storage:

        attr_df = {}

        for a in attributes:
            try:
                attr_df[a] = storage.load(label_files[a])
            except:
                continue

        for i in tqdm(range(len(issues))):
            issue = issues.iloc[i]

            stored_attrs = {
                a: issue.id in (attr_df[a]["id"].values if a in attr_df else [])
                for a in attributes
            }
            # check if all attributes are resolved
            if (
                # len(attributes) == len(attr_df) and
                any(stored_attrs[a] for a in stored_attrs)
            ):
                continue

            tqdm.write(f"{bcolors.HEADER}Issue: {issue.id}{bcolors.ENDC}")
            tqdm.write(f"{bcolors.OKBLUE}Issue URL: {issue.issue_url}{bcolors.ENDC}")

            tqdm.write("-----------------------")
            if isinstance(issue.raw_post, str):
                tqdm.write(issue.raw_post)
            else:
                tqdm.write("No Post Content")
            tqdm.write("-----------------------")
            tqdm.write(f"{bcolors.OKCYAN}testing URL: {issue.test_url}{bcolors.ENDC}")

            if filterlists:
                tqdm.write(
                    f"{bcolors.OKGREEN}breaking filterlist path: {filterlists.joinpath(f'{issue.id}/before.txt')}{bcolors.ENDC}"
                )

            webbrowser.open(issue.test_url)

            error_val = input(
                "Any problems with the page? supported commands ['u' for update test url]. ex: u https://www.google.com\nAction: "
            )

            if len(error_val) > 2 and error_val[:2] == "u ":
                url = error_val[2:].replace(" ", "")

                # test_url_updates.urls_updates[issue.id] = url
                storage.save(
                    "manual-test-url.csv",
                    pd.DataFrame(
                        [
                            {"id": issue.id, "test_url": url},
                        ]
                    ),
                )

            att_vals = {}
            for attribute in attributes:
                if not stored_attrs.get(attribute, False):
                    att_val = input(f"{attribute}: ")
                    att_vals[attribute] = att_val

            for attribute in att_vals:
                if att_vals[attribute].strip():
                    storage.save(
                        label_files[attribute],
                        pd.DataFrame(
                            [
                                {"id": issue.id, attribute: att_vals[attribute]},
                            ]
                        ),
                    )

            tqdm.write("================================================")
            os.system("clear")


def main(program: str, args: List[str]):

    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument("-i", "--issues", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path, default=None)
    parser.add_argument("-f", "--filter", type=Path, default=None)
    parser.add_argument("-a", "--attrs", nargs="+", type=str, required=True)
    parser.add_argument("-s", "--start", type=int, default=None)
    parser.add_argument("-c", "--count", type=int, default=None)

    ns = parser.parse_args(args)

    out_fp = ns.out if ns.out is not None else Path(ns.issues).parent

    label(
        Path(ns.issues),
        out_fp,
        ns.attrs,
        ns.start,
        ns.count,
        filterlists=Path(ns.filter),
    )


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
