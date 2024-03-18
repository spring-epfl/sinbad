import pandas as pd
from pathlib import Path
from graph.database import Database

edits_df = pd.read_csv("./out-experiments/out-luis-naive-content-features/edits.csv")
exp_df = pd.read_csv("./crawl/datadir-luis/experiments.csv")
log_df = pd.read_csv("./out-experiments/out-luis-naive-content-features/log.csv")


with Database(Path("./crawl/datadir-luis/crawl-data.sqlite"), Path("./crawl/datadir-luis/experiments.csv")) as database:
    site_visits = database.sites_visits()


def test_issue(issue):

    print(site_visits[site_visits['issue_id'] == int(issue)].iloc[0])

    print("broken filterlist: ", f'./forums/ublock/data/filters/{issue}/before.txt')
    print("fixed filterlist: ", f'./forums/ublock/data/filters/{issue}/after.txt')
    print(f"issue url: https://github.com/uBlockOrigin/uAssets/issues/{issue}")
    

no_tree_issues = log_df[(log_df['node_count'] == 0) & (log_df["error"].isnull())]
no_tree_issue = no_tree_issues.merge(site_visits, on='issue_id', how='inner')

for _, issue in no_tree_issues.iterrows():
    print("-------------------------------------")
    test_issue(issue.issue_id)
    input("next issue?")



    