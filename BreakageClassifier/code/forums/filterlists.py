import argparse
from datetime import datetime
from io import TextIOWrapper
import json
import os
from re import T, split
import sys
from pathlib import Path
import time
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


class BadIssueException:
    pass


from BreakageClassifier.code.forums.utils import (
    BadFilterlistSource,
    get_commit_date,
    dump_static_filterlist,
    headers,
    get_latest_commit_until,
    dump_list_from_commit,
)

tqdm.pandas()


def read_metadata(fp: Path) -> set:
    # in this metadata file we keep track of the added filterlists

    # add .meta
    meta_fp = fp.parent / (fp.name + ".meta")

    if meta_fp.exists():
        # each line is the name of a filterlist
        with open(meta_fp) as f:
            return {line.strip() for line in f.readlines()}
    else:
        return {}


def write_metadata(fp: Path, filterlists: set, append: bool = False):
    meta_fp = fp.parent / (fp.name + ".meta")

    if append and meta_fp.exists():
        with open(meta_fp, "a") as f:
            for fl in filterlists:
                f.write(fl + "\n")
    else:
        with open(meta_fp, "w") as f:
            for fl in filterlists:
                f.write(fl + "\n")


def get_directories_from_issue(
    issue: pd.Series,
):
    files: List[str] = json.loads(issue.directories)

    dirs = set()

    for file in files:
        if file.endswith(".txt"):
            dirs.add("/" + file.rsplit("/", 1)[0])

        elif "." in file:
            continue
        else:
            dirs.add("/" + file)

    return list(dirs)


def get_filterlist_from_post(
    fl_info: dict,
    output_path: Path,
    commit_dict: Optional[dict] = None,
    until: Optional[datetime] = None,
    append: bool = False,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    assert (
        commit_dict is not None or until is not None
    ), "Either commit_sha or until must be provided"

    lines = []

    existing_filterlists = read_metadata(output_path)
    added_filterlists = set()

    with open(output_path, "a" if append else "w") as f:
        f.write("\n! ---------context ------ \n")

        for li in fl_info:
            li_info = fl_info[li]

            if li in existing_filterlists:
                continue

            if "repo" in li_info:
                if until is None:
                    commit = commit_dict[li]
                else:
                    commit = get_latest_commit_until(li_info["repo"], until)

                f.write(f"# {li}\n")

                lines.append(
                    dump_list_from_commit(
                        f,
                        li_info["repo"],
                        commit["sha"],
                        li_info.get("dirs", []),
                        li_info.get("files", []),
                    )
                )

                added_filterlists.add(li)

            elif "static" in li_info:
                f.write(f"# {li}: static @ {li_info['static']}\n")
                try:
                    lines.append(dump_static_filterlist(f, li_info["static"]))
                except BadFilterlistSource as e:
                    tqdm.write(f"Error: {e}")
                    continue

                added_filterlists.add(li)

            else:
                f.write(f"# {li}\n")
                f.write(f"# List {li_info}\n")
                tqdm.write(f"List {li} {li_info}\n")

    write_metadata(output_path, added_filterlists, append=append)
    tqdm.write(f"Filterlist {output_path} added {sum(lines)} lines")

    return lines


def get_directories(repo: str, commit: str) -> List[str]:
    """gets the paths of relevant filterlists from a commit to a repo

    Args:
        repo (str): the repository id
        commit (str): the commit SHA

    Returns:
        List[str]: the list of filterlist paths
    """
    if str(repo) == "nan":
        return []

    req = f"https://api.github.com/repos/{repo}/commits/{commit}"
    resp = requests.get(req, headers=headers).json()

    directories = set()
    for file in resp["files"]:
        *directory, _ = file["filename"].split("/")
        directory = "/".join(directory)
        directories.add(directory)

    return list(directories)


def get_reverse_exception(rule):
    if rule:
        rules = []

        for line in rule.split("\n"):
            if line[:2] == "||":
                rules.append(line.replace("||", "@@||"))
            if line[:4] == "@@||":
                rules.append(line.replace("@@||", "||"))
            if "#@#" in line:
                rules.append(line.replace("#@#", "##"))

        return "\n".join(rules)
    return ""


def get_rules_from_commit_patch(patch: str) -> str:
    before_rules = []
    after_rules = []

    # if the line starts with @@,
    # if the line starts with a +, it's an addition unless it's after a - it's a change
    # if the line starts with a -, it's a removal

    lines = patch.split("\n")

    for i, line in enumerate(lines):
        if line.startswith("@@"):
            continue

        if line.startswith("+"):
            after_rules.append(line[1:])

        if line.startswith("-"):
            before_rules.append(line[1:])

    return "\n".join(before_rules), "\n".join(after_rules)


def generate_fl_info(filter_names: list, context: dict) -> dict:
    fl_info = {}
    for fl_name in filter_names:
        if fl_name not in context:
            print(fl_name)
            raise BadIssueException(f"Filterlist {fl_name} not found in context")
        fl_info[fl_name] = context[fl_name]
    return fl_info


def generate_filterlist(
    row: pd.Series,
    out: Path,
    context: Optional[dict] = None,
    force: bool = False,
):
    """compiles filterlist from the files of a commit

    Args:
        row (pd.Series): the information about a sample from the dataset
        out (Path): path to the folder containing the filterlists
    """

    out_path = Path(f"{out}/{row.id}")
    tqdm.write(str(out_path))

    # if (
    #     not force
    #     and out_path.exists()
    #     and out_path.joinpath("before.txt").is_file()
    #     and out_path.joinpath("after.txt").is_file()
    #     and os.stat(out_path.joinpath("before.txt")).st_size > 0
    #     and os.stat(out_path.joinpath("after.txt")).st_size > 0
    # ):
    #     return

    repo = row.repo

    # before commit is maybe very new compared to the issue date
    # before_commit = row.before_commit
    before_commit_data = get_latest_commit_until(
        repo, datetime.fromisoformat(row.created_at)
    )
    before_commit = before_commit_data["sha"]
    after_commit = row.after_commit
    changed_file_paths = json.loads(row.directories)

    out_path.mkdir(parents=True, exist_ok=True)

    with open(f"{out}/{row.id}/before.txt", "w") as f:
        f.write("! ---- before change filterlist ----- \n")
        dump_list_from_commit(
            f,
            repo,
            before_commit,
            [],
            changed_file_paths,
        )

        f.write("! ---- before change ----- \n")
        f.write(row.before_rule)
        f.write("! ------------------------ \n")

    with open(f"{out}/{row.id}/after.txt", "w") as f:
        f.write("! ---- after change filterlist ----- \n")
        dump_list_from_commit(
            f,
            repo,
            after_commit,
            [],
            changed_file_paths,
        )

        f.write("! ---- after change ----- \n")
        f.write(row.after_rule)
        f.write("! ------------------------ \n")

    # if we are adding the filterlists from the dataset context

    if context is not None and "filters" in row:
        # before.txt
        get_filterlist_from_post(
            generate_fl_info(json.loads(row["filters"]), context),
            output_path=out_path / "before.txt",
            until=datetime.fromisoformat(row["created_at"]),
            append=True,
        )

        # after.txt
        get_filterlist_from_post(
            generate_fl_info(json.loads(row["filters"]), context),
            output_path=out_path / "after.txt",
            until=get_commit_date(repo, after_commit),
            append=True,
        )


def generate_filterlists(
    issues_fp: Path, out_fp: Path, context_fp: Optional[Path] = None, force=False
):
    df = pd.read_csv(str(issues_fp.resolve()))
    df = df[df["should_include"]]

    # processing issues with less directories first
    # the dataframe list column is ex: ['a', 'b', 'c'] --> json fails to parse this as a list
    # so we need to convert it to a list of strings first
    df.sort_values(
        by="created_at",
        # key=lambda x: x.apply(lambda x: len(json.loads(x))),
        inplace=True,
        ascending=False,
    )

    context = None

    if context_fp is not None:
        with open(context_fp) as f:
            context = json.load(f)

    issues_with_errors = []

    for _, issue in tqdm(df.iterrows(), total=len(df)):
        try:
            generate_filterlist(
                issue,
                str(out_fp.joinpath("filterlists").resolve()),
                force=force,
                context=context,
            )
        except Exception as e:
            tqdm.write(f"Error: {e}")
            tqdm.write(f"Issue: {issue}")
            issues_with_errors.append(issue)
            continue

    while len(issues_with_errors) > 0:
        curr_issue = issues_with_errors.pop(0)
        try:
            generate_filterlist(
                curr_issue,
                str(out_fp.joinpath("filterlists").resolve()),
                force=force,
                context=context,
            )
        except Exception as e:
            tqdm.write(f"Error: {e}")
            tqdm.write(f"Issue: {curr_issue}")
            issues_with_errors.append(curr_issue)
            continue

        time.sleep(10 * 60)


def main(program: str, args: List[str]):
    """main routine"""
    parser = argparse.ArgumentParser(prog=program)
    parser.add_argument(
        "--input", type=Path, default=Path("./ublock/data/ublock-data.csv")
    )

    parser.add_argument("--out", type=Path, default=Path("./ublock/data"))
    parser.add_argument("--context", type=Path, default=None)
    parser.add_argument("--force", type=bool, default=False)

    ns = parser.parse_args(args)

    generate_filterlists(Path(ns.input), Path(ns.out), Path(ns.context), ns.force)


from adblockparser import AdblockRules


def read_file_newline_stripped(fname):
    with open(fname) as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines


def create_filterlist_rules(filterlist_path):
    rule_dict = {}
    rules = read_file_newline_stripped(filterlist_path)
    rule_dict["script"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["script", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["script_third"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["third-party", "script", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["image"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["image", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["image_third"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["third-party", "image", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["css"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["stylesheet", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["css_third"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["third-party", "stylesheet", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["xmlhttp"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["xmlhttprequest", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["xmlhttp_third"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["third-party", "xmlhttprequest", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["third"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["third-party", "domain", "subdocument"],
        skip_unsupported_rules=False,
    )
    rule_dict["domain"] = AdblockRules(
        rules,
        use_re2=False,
        max_mem=1024 * 1024 * 1024,
        supported_options=["domain", "subdocument"],
        skip_unsupported_rules=False,
    )

    return rule_dict


def match_url(domain_top_level, current_domain, current_url, resource_type, rules_dict):
    try:
        if domain_top_level == current_domain:
            third_party_check = False
        else:
            third_party_check = True

        if resource_type == "sub_frame":
            subdocument_check = True
        else:
            subdocument_check = False

        if resource_type == "script":
            if third_party_check:
                rules = rules_dict["script_third"]
                options = {
                    "third-party": True,
                    "script": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["script"]
                options = {
                    "script": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif resource_type == "image" or resource_type == "imageset":
            if third_party_check:
                rules = rules_dict["image_third"]
                options = {
                    "third-party": True,
                    "image": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["image"]
                options = {
                    "image": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif resource_type == "stylesheet":
            if third_party_check:
                rules = rules_dict["css_third"]
                options = {
                    "third-party": True,
                    "stylesheet": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["css"]
                options = {
                    "stylesheet": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif resource_type == "xmlhttprequest":
            if third_party_check:
                rules = rules_dict["xmlhttp_third"]
                options = {
                    "third-party": True,
                    "xmlhttprequest": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }
            else:
                rules = rules_dict["xmlhttp"]
                options = {
                    "xmlhttprequest": True,
                    "domain": domain_top_level,
                    "subdocument": subdocument_check,
                }

        elif third_party_check:
            rules = rules_dict["third"]
            options = {
                "third-party": True,
                "domain": domain_top_level,
                "subdocument": subdocument_check,
            }

        else:
            rules = rules_dict["domain"]
            options = {"domain": domain_top_level, "subdocument": subdocument_check}

        return rules.should_block(current_url, options)

    except Exception as e:
        print("Exception encountered", e)
        print("top url", domain_top_level)
        print("current url", current_domain)
        return False


def split_filterlist_file(filterlist_path: Path, lines_per_file: int):
    # split /18930/after.txt into /18930/after_1.txt, /18930/after_2.txt, etc.

    # read the file
    with open(filterlist_path) as f:
        i = 0

        lines = 0
        fw = open(filterlist_path.parent / f"{filterlist_path.stem}_{i}.txt", "w")

        for line in f:
            fw.write(line)
            lines += 1
            if lines == lines_per_file:
                i += 1
                lines = 0
                fw.close()
                fw = open(
                    filterlist_path.parent / f"{filterlist_path.stem}_{i}.txt", "w"
                )

        fw.close()


def blocked_requests(requests_df: pd.DataFrame, filterlist_path: Path):
    filterlist_rules = create_filterlist_rules(filterlist_path)

    blocked_ = requests_df.swifter.apply(
        lambda x: match_url(
            x.top_level_domain,
            x.current_domain,
            x.url,
            x.resource_type,
            filterlist_rules,
        ),
        axis=1,
    )
    return blocked_


if __name__ == "__main__":
    main(sys.argv[0], sys.argv[1:])

    # __dir__ = Path(__file__).parent

    # split_filterlist_file(__dir__ / "./adguard/data/filterlists/163836/before.txt", 10000)
