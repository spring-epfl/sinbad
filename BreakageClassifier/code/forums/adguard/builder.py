from collections import namedtuple
from datetime import datetime
import json
from pathlib import Path
import re
import traceback
import pandas as pd
import requests
import sys, os
import argparse

from storage_dump.storage import DataframeCSVStorageController
from tqdm import tqdm

from BreakageClassifier.code.forums.filterlists import get_rules_from_commit_patch
from BreakageClassifier.code.forums.language import detect_language

__dir__ = Path(__file__).parent.absolute()

sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))

from storage_dump.json_storage import JSONStorageController
from BreakageClassifier.code.forums.utils import (
    get_commit_date,
    get_commit_parent,
    get_repo_and_commit_from_url,
    get_commit_files,
    headers,
)

CommitGroup = namedtuple("CommitGroup", ["repo", "commit", "date"])

with open(__dir__ / "lists.json", "r") as f:
    FILTER_LIST_DEFS = json.load(f)


def crawl_issues(
    out_dir: Path,
    is_breaking= True
):
    if is_breaking:
        URL_PATTERN = "https://api.github.com/repos/AdguardTeam/AdguardFilters/issues?labels=T: Incorrect Blocking,A: Resolved&state=closed&per_page=100&page={}"
    else:
        URL_PATTERN = "https://api.github.com/repos/AdguardTeam/AdguardFilters/issues?labels=T: Ads,A: Resolved&state=closed&per_page=100&page={}"
    # pattern:
    # - label: T: Incorrect Blocking, A: Resolved
    # - state: closed
    # - per_page: 100
    # - page: 0, 1, 2, ...
    # - sort: most recent first

    with JSONStorageController(
        output_dir=out_dir, filenames=["issues.json", "failed.json"], json_mode="list"
    ) as storage:
        page = 0

        # handle pagination
        while True:
            resp = requests.get(URL_PATTERN.format(page), headers=headers)

            if resp.status_code != 200:
                print("Failed to fetch page {}".format(page))
                print(resp.text)

                storage.save(
                    "failed.json",
                    [
                        {
                            "page": page,
                            "url": URL_PATTERN.format(page),
                            "error": "Failed to fetch page",
                            "status_code": resp.status_code,
                            "content": resp.content,
                        },
                    ],
                )

                page += 1
                continue

            if not isinstance(resp.json(), list):
                print("Invalid response")

                storage.save(
                    "failed.json",
                    [
                        {
                            "page": page,
                            "url": URL_PATTERN.format(page),
                            "error": "Invalid response",
                            "status_code": resp.status_code,
                            "content": resp.content,
                        },
                    ],
                )

                page += 1
                continue

            if len(resp.json()) == 0:
                print("No more issues")

                break

            storage.save(
                "issues.json",
                resp.json(),
            )

            page += 1


URL_REGEX = r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'


def extract_test_url(body: str):
    test_issue_title = [
        "### Issue URL",
        "**Issue URL",
        "Where is the problem encountered?",
    ]

    # extract the test url from these lines
    # ### Where is the problem encountered?\n\nedgeservices.bing.com
    # ### Issue URL (Incorrect Blocking)\r\n [https://pixysos.com/flame]
    lines = body.split("\n")
    for i, line in enumerate(lines):
        if any([title in line for title in test_issue_title]):
            for j in range(i, len(lines)):
                # extract the url from the next lines
                matches = re.findall(URL_REGEX, lines[j])

                if len(matches) == 0:
                    continue

                return matches[0]

            raise Exception(f"Failed to extract test url from {body}")

    return None


def check_events_for_commits(issue_id):
    events_url = f"https://api.github.com/repos/AdguardTeam/AdguardFilters/issues/{issue_id}/events"
    events_resp = requests.get(events_url, headers=headers).json()
    if not isinstance(events_resp, list):
        raise Exception(f"Invalid response from {events_url}: {events_resp}")

    return [
        CommitGroup(
            *get_repo_and_commit_from_url(event["commit_url"]),
            datetime.strptime(event["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
        )
        for event in events_resp
        if event["commit_id"] is not None
    ]


def check_comments_for_commits(issue_id):
    comments_url = f"https://api.github.com/repos/AdguardTeam/AdguardFilters/issues/{issue_id}/comments"
    comments_resp = requests.get(comments_url, headers=headers).json()

    if not isinstance(comments_resp, list):
        raise Exception(f"Invalid response from {comments_url}: {comments_resp}")

    commits = []

    for comment in comments_resp:
        if comment["author_association"] != "CONTRIBUTOR":
            continue

        urls = re.findall(pattern=URL_REGEX, string=comment["body"])

        for url in filter(lambda x: "/commit/" in x, urls):
            repo, commit = get_repo_and_commit_from_url(url)

            commits.append(
                CommitGroup(
                    repo,
                    commit,
                    datetime.strptime(comment["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
                )
            )

    return commits


def crawl_commits(issue_id):
    commits = check_events_for_commits(issue_id)

    if len(commits) == 0:
        commits = check_comments_for_commits(issue_id)

    if len(commits) == 0:
        return None, None

    first_commit = commits[0]
    last_commit = commits[-1]

    par_commit_sha = get_commit_parent(first_commit.repo, first_commit.commit)
    par_commit = CommitGroup(
        first_commit.repo,
        par_commit_sha,
        get_commit_date(first_commit.repo, par_commit_sha),
    )

    if par_commit.repo != last_commit.repo:
        raise Exception(f"First commit and last commit are in different repos")

    return par_commit, last_commit


def get_files_data(commit_group: CommitGroup):
    files = get_commit_files(commit_group.repo, commit_group.commit)

    directories = []
    before_rule = ""
    after_rule = ""

    for file in files:
        directories.append(file["filename"])

        _before, _after = get_rules_from_commit_patch(file["patch"])

        before_rule += _before + "\n"
        after_rule += _after + "\n"

    return json.dumps(directories), before_rule, after_rule


def crawl_details(
    issues_fp: Path,
    force=False,
    date_threshold: datetime = None,
    count_threshold: int = None,
    count_fixed_threshold: int = None,
    debug=False,
):
    details_filename = "details.csv"
    unsolved_filename = "investigate.csv"

    # data prep
    data_dir = issues_fp.parent.resolve()

    with open(issues_fp, "r") as f:
        issues = json.load(f)

    max_i = min(len(issues), count_threshold) if count_threshold else len(issues)

    if force and os.path.exists(data_dir / details_filename):
        os.remove(data_dir / details_filename)

    with DataframeCSVStorageController(
        data_dir,
        [
            details_filename,
            unsolved_filename,
        ],
        is_async=False,
    ) as storage:
        # resuming a session
        try:
            details_df: pd.DataFrame = storage.load(details_filename)
        except FileNotFoundError:
            force = True

        processed_issue_ids = set(details_df["id"].unique()) if not force else set()
        count = 0
        count_fixed = 0

        for issue in tqdm(issues[:max_i]):
            if issue["number"] in processed_issue_ids:
                continue

            if count_threshold and count >= count_threshold:
                break

            if count_fixed_threshold and count_fixed >= count_fixed_threshold:
                break

            post_create_date = datetime.strptime(
                issue["created_at"], "%Y-%m-%dT%H:%M:%SZ"
            )

            if date_threshold and post_create_date < date_threshold:
                break

            row_out = {
                "id": issue["number"],
                "title": issue["title"],
                "created_at": post_create_date,
                "test_url": None,
                "issue_url": issue["html_url"],
                "repo": None,
                "before_commit": None,
                "after_commit": None,
                "directories": None,
                "raw_post": None,
                "lang": None,
                "should_include": True,
                "processed": False,
            }

            # process the issue

            row_out["lang"], row_out["raw_post"] = detect_language(issue["body"])
            try:
                row_out["test_url"] = extract_test_url(issue["body"])

                if row_out["test_url"] is None:
                    row_out["should_include"] = False

                par_commit, after_commit = crawl_commits(issue["number"])

                if par_commit is None or after_commit is None:
                    row_out["should_include"] = False

                else:
                    row_out["repo"] = par_commit.repo
                    row_out["before_commit"] = par_commit.commit
                    row_out["after_commit"] = after_commit.commit
                    (
                        row_out["directories"],
                        row_out["before_rule"],
                        row_out["after_rule"],
                    ) = get_files_data(after_commit)

                    row_out["processed"] = True

                    storage.save(details_filename, pd.DataFrame([row_out]))

            except Exception as e:
                trace = traceback.format_exc()
                row_out["should_include"] = False
                row_out["processed"] = False

                storage.save(
                    unsolved_filename,
                    pd.DataFrame([row_out | {"error": str(e), "trace": trace}]),
                )

                tqdm.write("Failed to process issue {}: {}".format(issue["number"], e))


class BadIssueException(Exception):
    pass


olf_filter_names = {
    "Peter Lowe's list": "Peter Lowe's Blocklist",
    "Fanboy's Cookiemonster List": "EasyList Cookie List",
    "Korean Adblock List": "List-KR",
    "AdGuard Annoyances": "AdGuard Annoyances filter",
    "AdGuard Popups": "AdGuard Popups filter",
    "Malware Domains Blocklist": "Online Malicious URL Blocklist",
    "Fanboy's Vietnamese": "Vietnamese adblock filter list",
    "AdGuard URL Tracking": "AdGuard URL Tracking filter",
    "AdGuard Mobile Ads": "AdGuard Mobile App Banners filter",
    "AdGuard Mobile App Banners": "AdGuard Mobile App Banners filter",
    "Fanboy Anti-Facebook List": "Fanboy's Anti-Facebook List",
    "AdGuard Base": "AdGuard Base filter",
    "FanboyEspanol": "EasyList Spanish",
    "AdGuard Turkish": "AdGuard Turkish filter",
    "AdGuard Ukrainian": "AdGuard Ukrainian filter",
    "AdGuard Simplified domain names": "AdGuard DNS filter",
    "AdGuard Cookie Notices": "AdGuard Cookie Notices filter",
    "ROLIST": "ROList",
    "AdGuard Other Annoyances": "AdGuard Other Annoyances filter",
    "AdGuard French": "AdGuard French filter",
    "AdGuard Widgets": "AdGuard Widgets filter",
    "AdGuard Tracking Protection": "AdGuard Tracking Protection filter",
    "AdGuard Dutch": "AdGuard Dutch filter",
    "AdGuard Experimental": "AdGuard Experimental filter",
    "Easylist Spanish": "EasyList Spanish",
    "AdGuard Japanese": "AdGuard Japanese filter",
    "AdGuard Spanish/Portuguese": "AdGuard Spanish/Portuguese filter",
    "AdGuard DNS": "AdGuard DNS filter",
    "AdGuard Russian": "AdGuard Russian filter",
    "AdGuard German": "AdGuard German filter",
    "AdGuard Chinese": "AdGuard Chinese filter",
    "AdGuard Social Media": "AdGuard Social Media filter",
    "Hungarian": "Hungarian filter",
    "Fanboy's Spanish/Portuguese": "EasyList Portuguese",
    "Thai Ads Filters": "EasyList Thailand",
}


def _extract_filters_from_system_config(post: str):
    filters = []
    sys_conf_i = post.find("System configuration")
    filters_i = post.find("Filters:", sys_conf_i)

    if filters_i == -1:
        raise BadIssueException("No filters found in issue")

    lines = post[filters_i:].split("\n")

    if len(lines) > 1 and ":" not in lines[1]:
        raise BadIssueException("Filter of issue Not As Expected")

    filters_text = lines[0].split(" | ")[1]

    # remove everything between <b> ... </b> and what is in them

    b_ptrn = r"<b>(.*?)</b>"

    filters_text = re.sub(b_ptrn, "", filters_text)

    # split by <br/> and remove empty strings

    filters = list(filter(lambda x: len(x), filters_text.split("<br/>")))

    # remove remaining tags and \r
    rem_ptrn = r"<(.*)>|<(.*)/>|\r"

    filters_out = []

    for _filter in filters:
        _filter = re.sub(rem_ptrn, "", _filter)

        # split by comma and remove empty strings
        filters_in_line = _filter.strip().split(",")

        filters_in_line = list(map(lambda x: x.strip(), filters_in_line))

        filters_in_line = list(filter(lambda x: len(x), filters_in_line))

        filters_out.extend(filters_in_line)

    # substitute old names
    filters_out = list(
        map(lambda x: olf_filter_names[x] if x in olf_filter_names else x, filters_out)
    )

    if not all(x in FILTER_LIST_DEFS for x in filters_out):
        bad_filters = set(filters_out) - set(FILTER_LIST_DEFS.keys())

        raise BadIssueException(
            "Issue contains unknown filters: {}".format(bad_filters)
        )

    return filters_out


def _extract_filters_from_filters_enabled(post: str):
    question_i = post.find("What filters do you have enabled?")

    lines = post[question_i:].split("\n")

    index = 1

    while lines[index].strip() == "":
        index += 1

    filters_out = list(
        filter(lambda x: len(x), map(lambda x: x.strip(), lines[index].split(",")))
    )

    if len(filters_out) == 0:
        raise BadIssueException("No filters found in issue: {}".format(lines[1]))

    # substitute old names
    filters_out = list(
        map(lambda x: olf_filter_names[x] if x in olf_filter_names else x, filters_out)
    )

    if not all(x in FILTER_LIST_DEFS for x in filters_out):
        bad_filters = set(filters_out) - set(FILTER_LIST_DEFS.keys())

        raise BadIssueException(
            "Issue contains unknown filters: {}".format(bad_filters)
        )

    return filters_out


def extract_filter_names(post: str):
    # if bad structure for filterlist names, raise exception
    if all(
        x not in post
        for x in ["System configuration", "What filters do you have enabled?"]
    ):
        raise BadIssueException("No headers found")

    filter_names = set()

    if "System configuration" in post:
        filter_names.update(_extract_filters_from_system_config(post))

    else:
        filter_names.update(_extract_filters_from_filters_enabled(post))

    return filter_names


def extract_filter_names_from_issues(
    details_fp: Path,
):
    details_df = pd.read_csv(details_fp)

    filter_names_out = []

    for _, issue in tqdm(details_df.iterrows(), total=len(details_df)):
        try:
            filters = list(extract_filter_names(issue["raw_post"]))
            filter_names_out.append(
                {
                    "id": issue["id"],
                    "filters": json.dumps(filters),
                    "success": True,
                    "error": None,
                }
            )
        except BadIssueException as e:
            filter_names_out.append(
                {
                    "id": issue["id"],
                    "filters": [],
                    "success": False,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(filter_names_out)

    df.to_csv(details_fp.parent / "filter-names.csv", index=False)

    print(f"Succeeded: {len(df[df['success'] == True])}")
    print(f"Failed: {len(df[df['success'] == False])}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(__dir__ / "data"))
    parser.add_argument("--neg", type=bool, default=False)
    args = parser.parse_args()

    # crawl_issues(
    #     Path(args.out),
    #     not args.neg
    # )

    # crawl_details(
    #     __dir__ / "adguard-neg" / "issues.json",
    # )

    extract_filter_names_from_issues(
        __dir__ / "adguard-neg" / "details.csv",
    )
