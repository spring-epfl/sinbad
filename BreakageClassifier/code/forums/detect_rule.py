import json
from pathlib import Path
import re
from types import FunctionType
import pandas as pd
import requests
from BreakageClassifier.code.forums.utils import headers
from tqdm import tqdm
from storage_dump.storage import DataframeCSVStorageController

# https://api.github.com/repos/easylist/easylist/commits/921e33ff49fca97b752fbc01923a3fadd4303d31

tqdm.pandas()


class RuleType:
    NETWORK = "NET"
    CSS = "CSS"
    INCONCLUSIVE = "XXX"
    ERROR = "ERR"


def judge_rule_type(rule: str):
    if any(a in rule for a in ["##", "#@#", "#?#"]):
        return RuleType.CSS

    return RuleType.NETWORK


def generate_rule_type(issue: pd.Series, on_end: FunctionType = None) -> str:
    rtype = None

    try:
        req = f"https://api.github.com/repos/{issue.repo}/commits/{issue.after_commit}"

        resp = requests.get(req, headers=headers)
        resp = resp.json()

        if "files" not in resp:
            tqdm.write("ERROR: no files in response")
            print(resp)
            raise Exception("no files in response")

        preds = set([judge_rule_type(file["patch"]) for file in resp["files"]])

        if len(preds) == 1:
            rtype = list(preds)[0]

        else:
            for file in resp["files"]:
                tqdm.write(
                    file["patch"][: min(len(file["patch"]), 30)].replace("\n", " ")
                )
            rtype = RuleType.INCONCLUSIVE

    except Exception as e:
        tqdm.write("ERROR:" + str(e))
        rtype = RuleType.ERROR

    s = pd.Series({"id": issue.id, "rule": rtype})

    if on_end:
        on_end(s)

    return s


def generate_rule_types(issues: pd.DataFrame, on_new: FunctionType = None):
    return issues.progress_apply(lambda x: generate_rule_type(x, on_new), axis=1)


def generate_rule_types_from_file(issues_fp: Path, out_fp: Path = None):
    issues = pd.read_csv(issues_fp.resolve())
    issues = issues[issues["should_include"]]
    
    print(headers)
    

    if out_fp is None:
        out_fp = issues_fp.parent

    with DataframeCSVStorageController(
        out_fp,
        [
            "rule-types01.csv",
        ],
    ) as storage:

        def store(s):
            storage.save(
                "rule-types01.csv",
                pd.DataFrame(
                    [
                        s,
                    ]
                ),
            )
            
        out_df = issues.progress_apply(lambda x: generate_rule_type(x, store), axis=1)

    return out_df
