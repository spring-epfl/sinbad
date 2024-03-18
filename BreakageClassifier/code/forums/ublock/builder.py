"""
The following code is adapted from Luis Vargas Work on forum scraping
https://github.com/spring-epfl/spring22-LuisVargas
"""

import argparse
import csv
from datetime import datetime
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

import requests
from storage_dump.storage import DataframeCSVStorageController
from tqdm import tqdm
import yaml

sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], "../.."))
sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))


from BreakageClassifier.code.forums.utils import URL_REGEX, get_commit_date, get_repo_and_commit_from_url, headers
import BreakageClassifier.code.forums.utils
from language import detect_language

__dir__ = os.path.dirname(os.path.abspath(__file__))

with open(__dir__ + "/lists.json", "r") as f:
    LISTS = json.load(f)


class BadPostException(Exception):
    pass


def get_filterlists_from_post(post: dict):
    if "body" not in post:
        raise BadPostException("Post does not have a body")

    body = post["body"]

    start_pattern = r"Configuration[.\s\S\n\r]*```yaml[\n\r]*"
    group_pattern = r"Configuration[.\s\S\n\r]*```yaml[\n\r]*[.\s\S]*(?=```)"

    matches = re.findall(group_pattern, body, re.MULTILINE)

    # should only be one match
    if len(matches) > 1:
        raise BadPostException(
            "More than one configuration found in post: \n" + "\n\n".join(matches)
        )

    if len(matches) == 0:
        raise BadPostException("No configuration found in post: \n" + body)

    conf_yaml = matches[0]
    matches = re.findall(start_pattern, conf_yaml, re.MULTILINE)

    conf_yaml = conf_yaml[len(matches[0]) :]

    # remove * from the lines to make it loadable
    conf_yaml = conf_yaml.replace("*", "")

    conf = yaml.safe_load(conf_yaml)

    filters_dict = conf["listset (total-discarded, last updated)"]
    added_filters = list(filters_dict.get("added", {}).keys())
    default_filters = list(filters_dict.get("default", {}).keys())

    return added_filters + default_filters


def get_filterlist_info(uname: str):
    return LISTS.get(uname, {})


def generate_dataset(
    out_path: Path,
    in_data_path=Path("data_uassets.json"),
    in_manual_path=Path("manual_data.json"),
    force=False,
):
    """generates the dataset from the forum scrape

    Args:
        out (Path): the path to the output folder
    """

    processed_issues = set()

    if not force:
        try:
            df = pd.read_csv(out_path.joinpath("issues.csv"))

            print("Found issues.csv in directory. Generating incomplete issues...")
            processed_issues = set(df[~df["repo"].isna()]["id"].values)

        except:
            print("Generating from scratch...")

    with open(str(in_data_path.resolve())) as jsonFile:  # Get the info from the file
        data = json.load(jsonFile)

    with open(str(in_manual_path.resolve())) as jsonFile:
        manual_data = json.load(jsonFile)

    titles = []
    posts = []
    steps = []
    lang = []

    # Classification static (S), dynamic (D), none (N) and not valid (X). Done manually.
    classifications = manual_data["classifications"]
    categories = manual_data["categories"]

    # Steps to reproduce from every issue obtained manually.
    steps = manual_data["steps"]

    ids = []
    created_at = []

    valid_issues = []

    for i in range(
        len(data["issues"])
    ):  # Loop to get the body and the issues IDs from every issue
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            posts.append(str(data["issues"][i]["basic_info"]["body"]))
            ids.append(str(data["issues"][i]["issue_id"]))
            created_at.append(str(data["issues"][i]["basic_info"]["created_at"]))
            titles.append(str(data["issues"][i]["basic_info"]["title"]))

            valid_issues.append(data["issues"][i])

    data["issues"] = valid_issues

    # Code from Shuting to clean the OP
    for i in range(len(posts)):
        # remove URL contained in the post
        # URL regex patern credit to: https://gist.github.com/gruber/8891611
        pattern = (
            r"(?i)\b((?:(http|https)?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|"
            r"coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|"
            r"aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|"
            r"cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|"
            r"gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|"
            r"km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|"
            r"mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|"
            r"rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|"
            r"tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+"
            r"|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};"
            r':\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|'
            r"info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|"
            r"ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|"
            r"cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|"
            r"gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|"
            r"kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|"
            r"ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|"
            r"sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|"
            r"ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw|png|jpg)\b/?(?!@)))"
        )
        processed_post = re.sub(pattern, " ", steps[i])
        # remove digits contained in the post
        processed_post = re.sub("\d+", " ", processed_post)
        # remove other special character contained in the post
        processed_post = re.sub("\W+|_", " ", processed_post)
        processed_post = processed_post.strip()
        steps[i] = processed_post

    # Extract last patch for every issue
    patchs = []

    for i in range(len(data["issues"])):
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            try:
                if data["issues"][i]["commits"] and (
                    classifications[i] == "D" or classifications[i] == "S"
                ):
                    # patchs.append(data["issues"][i]["commit_0"]["files"][0]["patch"])
                    n = len(data["issues"][i]["commits"]) - 1
                    patchs.append(
                        data["issues"][i]["commits"][f"commit_{n}"]["files"][0]["patch"]
                    )
                else:
                    patchs.append("None")
            except KeyError:
                patchs.append("None")

    # Extract list updated
    filenames = []

    for i in range(len(data["issues"])):
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            try:
                if data["issues"][i]["commits"] and (
                    classifications[i] == "D" or classifications[i] == "S"
                ):
                    filenames.append(
                        data["issues"][i]["commits"][f"commit_{n}"]["files"][0][
                            "filename"
                        ]
                    )
                else:
                    filenames.append("None")
            except KeyError:
                filenames.append("None")

    print("Most common list changed:")
    print(Counter(filenames).most_common(5)[1:])

    # Obtain how rules change within time

    rules_before = []
    for i in range(len(data["issues"])):
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            try:
                if patchs[i] != "None":
                    if "\n+" in patchs[i].partition("\n-")[2]:
                        rules_before.append(
                            patchs[i].partition("\n-")[2].partition("\n+")[0]
                        )
                    elif "\n " in patchs[i].partition("\n-")[2]:
                        rules_before.append(
                            patchs[i].partition("\n-")[2].partition("\n ")[0]
                        )
                    else:
                        rules_before.append(patchs[i].partition("\n-")[2])
                else:
                    rules_before.append("None")
            except AttributeError:
                rules_before.append("None")

    rules_after = []
    for i in range(len(data["issues"])):
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            try:
                if patchs[i] != "None":
                    if "\n-" in patchs[i].partition("\n-")[2]:
                        rules_after.append(
                            patchs[i]
                            .partition("\n+")[2]
                            .partition("\n-")[0]
                            .partition("\n ")[0]
                        )
                    elif "\n " in patchs[i].partition("\n+")[2]:
                        rules_after.append(
                            patchs[i].partition("\n+")[2].partition("\n ")[0]
                        )
                    else:
                        rules_after.append(patchs[i].partition("\n+")[2])
                else:
                    rules_after.append("None")
            except AttributeError:
                rules_after.append("None")

    def get_commits(issue_id, _created_at: str):
        req = f"https://api.github.com/repos/uBlockOrigin/uAssets/issues/{issue_id}/events"
        resp = requests.get(req, headers=headers).json()
        commit_ids = []
        for event in resp:
            if event["commit_id"]:
                commit_ids.append(get_repo_and_commit_from_url(event["commit_url"]))

        if len(commit_ids) == 0:
            # check the comments
            req = f"https://api.github.com/repos/uBlockOrigin/uAssets/issues/{issue_id}/comments"
            resp = requests.get(req, headers=headers).json()
            commit_ids = []
            for comment in resp:
                urls = re.findall(pattern=URL_REGEX, string=comment["body"])

                for url in filter(lambda x: "/commit/" in x, urls):
                    commit_ids.append(get_repo_and_commit_from_url(url))

        commit_id = None
        first_commit_id = None

        if not len(commit_ids):
            return None, None, None, None, None

        repo_url, commit_id = commit_ids[-1]

        _created_at = datetime.strptime(_created_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=None
        )

        for repo, commit in commit_ids:
            if repo == repo_url and get_commit_date(repo, commit) > _created_at:
                first_commit_id = commit
                break

        if not commit_id:
            return None, None, None, None, None

        req = f"https://api.github.com/repos/{repo_url}/commits/{first_commit_id}"
        resp = requests.get(req, headers=headers).json()

        before_commit_id = resp["parents"][0]["sha"]
        before_commit_tree = resp["commit"]["tree"]["sha"]

        if commit_id != first_commit_id:
            req = f"https://api.github.com/repos/{repo_url}/commits/{commit_id}"
            resp = requests.get(req, headers=headers).json()

        directories = set()
        for file in resp["files"]:
            *directory, _ = file["filename"].split("/")
            directory = "/".join(directory)
            directories.add(directory)

        # get the files that must be queried
        req = f"https://api.github.com/repos/{repo_url}/git/trees/{before_commit_tree}?recursive=1"

        resp = requests.get(req, headers=headers).json()

        files = set()
        for file in resp["tree"]:
            if any([x in file["path"] for x in directories]):
                files.add(file["path"])

        return (
            repo_url,
            before_commit_id,
            commit_id,
            before_commit_tree,
            json.dumps(list(files)),
        )

    before_commits = []
    after_commits = []
    repos = []
    trees = []
    directories = []

    for i in tqdm(range(len(data["issues"]))):
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            try:
                rep, bef, aft, tree, dirs = get_commits(ids[i], created_at[i])
            except Exception as e:
                print(e)
                rep, bef, aft, tree, dirs = "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"

            before_commits.append(bef)
            trees.append(tree)
            after_commits.append(aft)
            repos.append(rep)
            directories.append(dirs)

    # In case something is empty lets put None
    for i in tqdm(range(len(data["issues"]))):
        if force or data["issues"][i]["issue_id"] not in processed_issues:
            rules_before[i] = rules_before[i].replace("\n-", "\n")
            rules_after[i] = rules_after[i].replace("\n+", "\n")
            if rules_before[i] == "":
                rules_before[i] = "None"
            if rules_after[i] == "":
                rules_after[i] = "None"

    # Merge and export to CSV just in case
    list_merge = list(
        zip(
            ids,
            titles,
            created_at,
            rules_before,
            rules_after,
            repos,
            before_commits,
            after_commits,
            directories,
            categories,
            classifications,
        )
    )

    Path(out_path).mkdir(parents=True, exist_ok=True)

    prev_data_exists = len(processed_issues) > 0

    with open(
        f"{out_path}/issues.csv", "w" if not prev_data_exists else "a", encoding="utf-8"
    ) as out_file:
        csv_out = csv.writer(out_file)

        if not prev_data_exists:
            csv_out.writerow(
                [
                    "id",
                    "title",
                    "created_at",
                    "before_rule",
                    "after_rule",
                    "repo",
                    "before_commit",
                    "after_commit",
                    "directories",
                    "category",
                    "label",
                ]
            )
        for row in list_merge:
            csv_out.writerow(row)


def generate_details(df_fp: Path, json_fp: Path, out_fp: Path):
    df = pd.read_csv(str(df_fp.resolve())).astype({"id": "int64"})

    with open(str(json_fp.resolve())) as jsonFile:  # Get the info from the file
        json_data = json.load(jsonFile)

    rows = []

    for i in range(len(json_data["issues"])):
        post = str(json_data["issues"][i]["basic_info"]["body"])
        id = str(json_data["issues"][i]["issue_id"])

        if "## Describe" in post and "## Screenshot" in post:
            i = post.find("## Describe")
            j = post.find("## Screenshot")

            post = "\n".join(post[i:j].split("\n")[1:])

            lang, clean_post = detect_language(post)

        else:
            lang = "no"
            clean_post = post

        rows.append(
            {
                "id": id,
                "raw_post": clean_post,
                "lang": lang,
                "issue_url": f"https://github.com/uBlockOrigin/uAssets/issues/{id}",
            }
        )

    df = df.merge(
        pd.DataFrame(rows).dropna().astype({"id": "int64"}), on="id", how="outer"
    )

    df = df.drop(columns=[c for c in df.columns if "unnamed" in c or "Unnamed" in c])

    df["should_include"] = ((df["label"] == "D") | (df["label"] == "S")) & ~df[
        "repo"
    ].isna()

    df.to_csv(str(out_fp.resolve()))

    return df


def get_n_commits(df_fp: Path, out_fp: Path):
    def get_commits(issue_id):
        def _get_repo_and_commit_from_url(url):
            if "api.github.com" in url:
                commits_sep = "/commits/"
                repo_sep = "https://api.github.com/repos/"

            else:
                commits_sep = "/commit/"
                repo_sep = "https://github.com/"

            repo_url, commit_id = url.split(commits_sep)
            _, repo_url = repo_url.split(repo_sep)

            return repo_url, commit_id

        req = f"https://api.github.com/repos/uBlockOrigin/uAssets/issues/{issue_id}/events"
        resp = requests.get(req, headers=headers).json()
        commit_ids = []
        for event in resp:
            if event["commit_id"]:
                commit_ids.append(_get_repo_and_commit_from_url(event["commit_url"]))

        return commit_ids, len(commit_ids)

    df = pd.read_csv(str(df_fp.resolve())).astype({"id": "int64"})

    with DataframeCSVStorageController(
        out_fp,
        [
            "commits.csv",
        ],
        is_async=False,
    ) as storage:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                commits, n_commits = get_commits(row.id)

                storage.save(
                    "commits.csv",
                    pd.DataFrame(
                        [
                            {"commits": commits, "n_commits": n_commits},
                        ]
                    ),
                )
            except Exception as e:
                print(e)


def main(program: str, args: List[str]):
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument("--out", type=Path, default=Path("./data"))

    ns = parser.parse_args(args)
    generate_dataset(ns.out)


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
