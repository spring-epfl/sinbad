import argparse
import json
from pathlib import Path
import re
import sys
from typing import List
import pandas as pd
import os

sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))

from BreakageClassifier.code.forums.utils import URL_REGEX

from pathlib import Path
import os

base_path = Path(os.path.join(os.path.split(os.path.abspath(__file__))[0]))


def semi_manual_extract_urls(out, is_force=False):

    urls = {}
    out_fp = f"{out}/out_urls.csv"

    if not is_force and os.path.isfile(out_fp):
        urls = pd.read_csv(f"{out}/out_urls.csv").to_dict()
        col1, col2 = list(urls.keys())
        urls = dict(zip(urls[col1].values(), urls[col2].values()))

    with open(
        str(base_path.joinpath("data_uassets.json").resolve()), "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    with open(out_fp, mode="a") as out_csv_file:

        for i, issue in enumerate(data["issues"]):

            if issue["issue_id"] in urls:
                print(
                    f"Already existing: issue {issue['issue_id']} url='{urls[issue['issue_id']]}'"
                )
                continue

            body = issue["basic_info"]["body"]

            url = None

            if body:
                section = body.split("URL address of the web page")

                if len(section) > 1:
                    section = section[1]
                    url = re.search(pattern=URL_REGEX, string=section)[0]

                if not url:
                    section = body.split("### URL(s) where the issue occurs")
                    if len(section) > 1:
                        section = section[1]

                        url = re.search(pattern=URL_REGEX, string=section)[0]

                if not url:
                    section = body.split("### Category")
                    if len(section) > 1:
                        section = section[0]
                        urls = re.findall(pattern=URL_REGEX, string=section)

                        title_url = re.search(
                            pattern=URL_REGEX, string=issue["basic_info"]["title"]
                        )[0]
                        for u in urls:
                            if title_url in u:
                                url = u

            if not url:
                url = re.search(pattern=URL_REGEX, string=issue["basic_info"]["title"])

                if url:
                    url = url[0]
                    if "http" not in url:
                        url = "https://" + url

            if not url:
                input("no url found")
                print(body)
                url = input("add url:")

            print(i, issue["issue_id"], issue["basic_info"]["title"], "\t", url)

            urls[issue["issue_id"]] = url

            if i == 0:
                _urls_pd = pd.Series(urls)
                _urls_pd.to_csv(out_fp)
            else:
                out_csv_file.write(f"{issue['issue_id']},\"{url}\"\n")

    # urls = pd.Series(urls)
    # urls.to_csv(out_fp)


def combine_df(out):
    urls = pd.read_csv(f"{out}/out_urls.csv", squeeze=True)
    df = pd.read_csv(f"{out}/ublock-data.csv")

    urls.columns = ["id", "url"]

    df = df.merge(urls, how="left", on="id")

    df.to_csv(f"{out}/ublock-data.csv")


def main(program: str, args: List[str]):
    """main routine"""
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument("--action", type=str, default="extract")
    parser.add_argument("--out", type=Path, default=Path("./data"))
    parser.add_argument("--force", type=bool, default=False)

    ns = parser.parse_args(args)

    if ns.action == "extract":
        semi_manual_extract_urls(ns.out, ns.force)

    elif ns.action == "combine":
        combine_df(ns.out)
    else:
        print("incorrect action: should be [extract/combine]")


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
