import traceback
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import os
from pathlib import Path

load_dotenv()
DOTENV_PATH = find_dotenv()

import BreakageClassifier.code.forums.utils as forum_utils
import BreakageClassifier.code.forums.filterlists as forum_filterlists


if __name__ == "__main__":
    issue_commits_df = pd.read_csv("../net_rep-out/easylist/issue-commits.csv")
    issues = pd.read_csv("../net_rep-out/easylist/easylist-data-rep.csv")
    
    issues.drop(columns= ['before_commit', 'after_commit'], inplace=True)
    
    df = issues.merge(issue_commits_df, on="id")

    failed = []

    for _, issue in tqdm(df.iterrows(), total=len(df)):
        out_dir = Path("../net_rep-out/easylist/filterlists-1") / str(issue.id)

        dirs = forum_filterlists.get_directories_from_issue(issue)
        
        # remove error log if it exists
        if os.path.exists(out_dir / "error.log"):
            os.remove(out_dir / "error.log")
            
        # if the files already exist, skip
        # if os.path.exists(out_dir / "before.txt") and os.path.exists(out_dir / "after.txt"):
        #     continue

        try:
            
            if not os.path.exists(out_dir / "before.txt"):
                
                forum_filterlists.get_filterlist_from_post(
                    {"easylist": {"repo": "easylist/easylist", "dirs": dirs}},
                    out_dir / "before.txt",
                    commit_dict={"easylist": {"sha": issue.before_commit, "date": None}},
                )
            
            
            forum_filterlists.get_filterlist_from_post(
                {"easylist": {"repo": "easylist/easylist", "dirs": dirs}},
                out_dir / "after_par.txt",
                commit_dict={"easylist": {"sha": issue.after_commit_par, "date": None}},
            )

            if not os.path.exists(out_dir / "after.txt"):
                forum_filterlists.get_filterlist_from_post(
                    {"easylist": {"repo": "easylist/easylist", "dirs": dirs}},
                    out_dir / "after.txt",
                    commit_dict={"easylist": {"sha": issue.after_commit, "date": None}},
                )

        except Exception as e:
            tr = traceback.format_exc()
            
            tqdm.write(tr)
            
            os.makedirs(out_dir, exist_ok=True)
            
            with open(out_dir / "error.log", "w") as f:
                f.write(tr)

            failed.append(issue.id)

    print("Failed: ", failed)
    print("total: ", len(issue_commits_df))
    print("failed: ", len(failed))
