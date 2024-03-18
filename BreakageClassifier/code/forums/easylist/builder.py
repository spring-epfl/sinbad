import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import traceback
from typing import List
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
import pandas as pd
import os

tqdm.pandas()

sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))

from BreakageClassifier.code.forums.utils import URL_REGEX, get_commit_date
from BreakageClassifier.code.forums.utils import headers as github_headers
from storage_dump.storage import DataframeCSVStorageController
from language import detect_language

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0"
}

EASY_LIST_ISSUES_URL = (
    "https://forums.lanik.us/viewforum.php?f=64-report-incorrectly-removed-content"
)
EASY_LIST_DOMAIN_URL = "https://forums.lanik.us"


def _get_repo_and_commit_from_url(url):

    split = url.split("/commit/")
    if len(split) != 2:
        split = url.split("/commits/")
    repo_url, commit_id = split
    _, repo_url = repo_url.split("https://github.com/")
    return repo_url, commit_id


def crawl_issues(out_dir):

    issues = []

    with DataframeCSVStorageController(
        Path(out_dir).resolve(),
        [
            "issues.csv",
        ],
        is_async=False,
    ) as out_storage:

        for i in tqdm(range(0, 7900, 50)):

            doc = requests.get(
                EASY_LIST_ISSUES_URL + "&start=" + str(i), headers=headers
            )
            doc = BeautifulSoup(doc.text, features="html.parser")
            doc = doc.find("ul", {"class": "topiclist topics"}).children

            _issues_page = []

            for j, li in enumerate(doc):

                if (
                    isinstance(li, Tag)
                    and "class" in li.attrs
                    and "row" in li.attrs["class"]
                ):
                    li_record = {}

                    # get title
                    a_title: Tag = li.find("a", {"class": "topictitle"})
                    li_record["issue_url"] = (
                        EASY_LIST_DOMAIN_URL + a_title.attrs["href"][1:]
                    )
                    li_record["id"] = (
                        a_title.attrs["href"][1:].split("?t=")[1].split("&sid=")[0]
                    )
                    li_record["title"] = a_title.text

                    # create date

                    create_date: Tag = li.select_one(".topic-poster time")
                    li_record["created_at"] = create_date.attrs["datetime"]

                    # replies
                    li_record["n_replies"] = int(
                        li.select_one(".posts").text.split()[0]
                    )
                    li_record["n_views"] = int(li.select_one(".views").text.split()[0])

                    # locked or not
                    _potential_locked_span = li.select_one(".prettytopic")
                    li_record["locked"] = (_potential_locked_span is not None) and (
                        "Locked" in _potential_locked_span.text
                    )

                    _issues_page.append(li_record)

                    # tqdm.write(
                    #     f"{i+j}: {i//50}/{j} - {li_record['title']} - {li_record['id']}"
                    # )

            _out = pd.DataFrame(_issues_page)
            out_storage.save("issues.csv", _out)
            issues.extend(_issues_page)


def crawl_details(
    list_fp,
    force=False,
    date_threshold: datetime = None,
    count_threshold=None,
    count_fixed_threshold=None,
    debug=False,
):
    def __crawl_issue(issue):
        EASY_LIST_ISSUE_URL = issue.issue_url
        doc = requests.get(EASY_LIST_ISSUE_URL, headers=headers)
        doc = BeautifulSoup(doc.text, features="html.parser")
        _contents = doc.select(".content")
        if not len(_contents):
            return (None, None, None, None, None, None, False, True, [])

        post_content = _contents[0]

        if post_content is None:
            return None, None, None, None, None, None, False, True, []

        urls = re.findall(pattern=URL_REGEX, string=post_content.text)
        urls_taken = [x for x in urls if issue.title.lower() in x.lower()]

        url = None
        should_include = True

        if len(urls_taken):
            url = urls_taken[0]

        if url is None and len(urls) == 0:
            url = issue.title

        if url is None:
            return None, None, None, None, None, post_content.text, False, False, urls

        if not should_include:
            return None, None, None, None, None, post_content.text, False, True, urls

        # get commits

        created_at = datetime.strptime(issue.created_at, "%Y-%m-%dT%H:%M:%S%z").replace(
            tzinfo=None
        )

        commits = []

        for post in doc.select(".post"):
            # if the .profile-rank exists and it has "EasyList Author" in it, skip
            if (
                post.select_one(".profile-rank")
                and "EasyList Author" in post.select_one(".profile-rank").text
            ):
                _post_content = post.select_one(".content")
                urls = [
                    _get_repo_and_commit_from_url(x.attrs["href"])
                    for x in _post_content.find_all("a")
                    if "github.com" in x.attrs["href"] and "commit" in x.attrs["href"]
                ]

                commits.extend(urls)

        commit_id = None
        first_commit_id = None

        repo_url = None
        if len(commits):
            repo_url, commit_id = commits[-1]

        for repo, commit in commits:
            if repo == repo_url and get_commit_date(repo, commit) > created_at:
                first_commit_id = commit
                break

        if not commit_id:
            return url, repo_url, None, None, None, post_content.text, False, True, []

        try:

            req = f"https://api.github.com/repos/{repo_url}/commits/{first_commit_id}"
            resp = requests.get(req, headers=github_headers).json()

            before_commit_id = resp["parents"][0]["sha"]
            before_commit_tree = resp["commit"]["tree"]["sha"]

        except Exception as e:
            if "message" in resp and resp["message"] == "Not Found":
                return (
                    url,
                    repo_url,
                    None,
                    None,
                    None,
                    post_content.text,
                    False,
                    False,
                    [],
                )
            else:
                raise e

        if commit_id != first_commit_id:
            req = f"https://api.github.com/repos/{repo_url}/commits/{commit_id}"
            resp = requests.get(req, headers=github_headers).json()

        directories = set()
        for file in resp["files"]:
            *directory, _ = file["filename"].split("/")
            directory = "/".join(directory)
            directories.add(directory)

        # get the files that must be queried
        req = f"https://api.github.com/repos/{repo_url}/git/trees/{before_commit_tree}?recursive=1"

        resp = requests.get(req, headers=github_headers).json()

        files = set()
        for file in resp["tree"]:
            if any([x in file["path"] for x in directories]):
                files.add(file["path"])

        return (
            url,
            repo_url,
            before_commit_id,
            commit_id,
            json.dumps(list(files)),
            post_content.text,
            True,
            True,
            [],
        )

    data_dir = Path(list_fp).parent.resolve()
    issues_df = pd.read_csv(list_fp)
    issues_df = issues_df[issues_df["locked"]]
    issues_df.sort_values(by="created_at", inplace=True, ascending=False)
    details_filename = "details.csv"
    unsolved_filename = "investigate.csv"

    count = 0
    count_fixed = 0

    with DataframeCSVStorageController(
        data_dir,
        [
            details_filename,
            unsolved_filename,
        ],
        is_async=False,
    ) as storage:
        try:
            details_df: pd.DataFrame = storage.load(details_filename)
        except:
            force = True

        for _, row in tqdm(
            issues_df.iterrows(),
            total=len(issues_df) if not count_threshold else count_threshold,
        ):

            if not force and row["id"] in set(details_df["id"]):
                continue

            if row["locked"] == False:
                continue

            if count_threshold and count >= count_threshold:
                break

            if count_fixed_threshold and count_fixed >= count_fixed_threshold:
                break

            post_create_date = datetime.strptime(
                row["created_at"].split("T")[0], "%Y-%m-%d"
            )

            if date_threshold and date_threshold > post_create_date:
                break

            row_out = {
                "id": row["id"],
                "test_url": None,
                "repo": None,
                "before_commit": None,
                "after_commit": None,
                "directories": None,
                "raw_post": None,
                "lang": None,
                "should_include": False,
                "processed": False,
            }

            try:

                test_url, repo, bc, ac, dire, raw_post, si, p, urls = __crawl_issue(row)

                if raw_post:
                    lang, clean_post = detect_language(raw_post)
                else:
                    lang = None
                    clean_post = None

                row_out["test_url"] = test_url
                row_out["repo"] = repo
                row_out["before_commit"] = bc
                row_out["after_commit"] = ac
                row_out["directories"] = dire
                row_out["raw_post"] = clean_post
                row_out["lang"] = lang
                row_out["should_include"] = si
                row_out["processed"] = p

                if si:
                    count_fixed += 1

                if len(urls) and not si:
                    storage.save(
                        unsolved_filename,
                        pd.DataFrame(
                            [
                                {
                                    "id": row["id"],
                                    "issue_url": row["issue_url"],
                                    "urls": urls,
                                }
                            ]
                        ),
                    )

            except Exception as e:
                traceback.print_exception(Exception, e, e.__traceback__)
                break

            storage.save(
                details_filename,
                pd.DataFrame(
                    [
                        row_out,
                    ]
                ),
            )

            count += 1

            if row_out["should_include"] and debug:
                post_gist = row_out["raw_post"].replace("\n", " ")[
                    : min(len(row_out["raw_post"]), 50)
                ]
                tqdm.write(f"Issue {row['id']}")
                tqdm.write(f"Title: {row['title']}")
                tqdm.write(f"Issue URL: {row['issue_url']}")
                tqdm.write(f"Testing URL: {row_out['test_url']}")
                tqdm.write(f"{row_out['lang']} content: {post_gist}...")
                tqdm.write(
                    f"after commit URL: https://api.github.com/repos/{row_out['repo']}/commits/{row_out['after_commit']}"
                )
                tqdm.write("================================")


def get_num_commits(
    list_fp,
    force=False,
    date_threshold: datetime = None,
    count_threshold=None,
):
    def __crawl_issue(issue):
        EASY_LIST_ISSUE_URL = issue.issue_url
        doc = requests.get(EASY_LIST_ISSUE_URL, headers=headers)
        doc = BeautifulSoup(doc.text, features="html.parser")
        _contents = doc.select(".content")
        if not len(_contents):
            return None
        post_content = _contents[0]

        if post_content is None:
            return None

        # get commits

        commits = []

        for i, post in enumerate(doc.select(".post")):
            if i == 0:
                continue

            _post_content = post.select_one(".content")
            urls = [
                _get_repo_and_commit_from_url(x.attrs["href"])
                for x in _post_content.find_all("a")
                if "github.com" in x.attrs["href"] and "commit" in x.attrs["href"]
            ]

            commits.extend(urls)

        return commits, len(commits)

    data_dir = Path(list_fp).parent.resolve()
    issues_df = pd.read_csv(list_fp)
    issues_df = issues_df[issues_df["locked"]]
    commits_filename = "commits.csv"

    with DataframeCSVStorageController(
        data_dir,
        [
            commits_filename,
        ],
        is_async=False,
    ) as storage:
        try:
            commits_df: pd.DataFrame = storage.load(commits_filename)
        except:
            force = True

        for _, row in tqdm(
            issues_df.iterrows(),
            total=len(issues_df) if not count_threshold else count_threshold,
        ):

            if not force and row["id"] in set(commits_df["id"]):
                continue

            if row["locked"] == False:
                continue

            post_create_date = datetime.strptime(
                row["created_at"].split("T")[0], "%Y-%m-%d"
            )

            if date_threshold and date_threshold > post_create_date:
                break

            row_out = {
                "id": row["id"],
                "commits": None,
                "n_commits": None,
            }

            try:

                commits, n_commits = __crawl_issue(row)

                row_out["commits"] = commits
                row_out["n_commits"] = n_commits

            except Exception as e:
                traceback.print_exception(Exception, e, e.__traceback__)
                break

            storage.save(
                commits_filename,
                pd.DataFrame(
                    [
                        row_out,
                    ]
                ),
            )


def combine(issues_fp: Path, details_fp: Path, out_fp: Path):

    issues_df = pd.read_csv(str(issues_fp.resolve()))
    details_df = pd.read_csv(str(details_fp.resolve()))

    dataset = issues_df.merge(
        details_df, on="id", left_index=False, right_index=False, how="left"
    )

    dataset.drop(columns=["processed", "Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)
    dataset.rename(
        columns={
            "locked": "fixed",
        },
        inplace=True,
    )
    dataset.loc[~dataset["should_include"].notna(), "should_include"] = False

    dataset.to_csv(str(out_fp.resolve()))


def main(program: str, args: List[str]):
    """main routine"""
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument(
        "--a", type=str, choices=["issues", "details", "combine"], default="issues"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("."), help="the output directory path"
    )
    parser.add_argument(
        "--issues", type=Path, default=None, help="the issues.csv filepath"
    )
    parser.add_argument(
        "--details", type=Path, default=None, help="the details.csv filepath"
    )
    parser.add_argument(
        "--force", type=bool, default=False, help="overwrite the details.csv file"
    )
    parser.add_argument(
        "--dt",
        type=str,
        default=None,
        help="scrape issues later than this date. format YYYY-MM-DD, ex: 2022-12-30",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="the maximum number of issue details to crawl.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=None,
        help="the maximum number of successful issue details to crawl.",
    )
    parser.add_argument("-v", type=bool, default=False, help="verbose mode.")

    ns = parser.parse_args(args)

    dt = None

    try:
        dt = datetime.strptime(ns.dt, "%Y-%m-%d")
    except:
        raise AssertionError("invalid date value for --dt")

    assert ns.n is None or ns.n > 0, "--n must be a positive integer"
    assert ns.ns is None or ns.ns > 0, "--ns must be a positive integer"

    if ns.a == "issues":
        crawl_issues(Path(ns.out))

    elif ns.a == "details":
        crawl_details(Path(ns.issues), ns.force, dt, ns.n, ns.ns, ns.v)

    elif ns.a == "combine":
        combine(
            Path(ns.issues),
            Path(ns.details),
            Path(ns.out).joinpath("easylist-data.csv"),
        )

    else:
        print(
            "Invalid --a. must be one of the following: 'issues', 'details', or 'combine'"
        )


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])
