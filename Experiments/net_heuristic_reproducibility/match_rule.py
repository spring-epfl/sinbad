import sqlite3
from pathlib import Path
import traceback
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import swifter
load_dotenv()
DOTENV_PATH = find_dotenv()
from storage_dump.storage import DataframeCSVStorageController
from BreakageClassifier.code.forums.filterlists import blocked_requests
from BreakageClassifier.code.graph.utils import get_domain

__dir__ = Path(__file__).parent


def get_requests(crawl_dir: Path):
    conn = sqlite3.connect(str(crawl_dir / "crawl-data.sqlite"))
    c = conn.cursor()

    experiments_df = pd.read_csv(crawl_dir / "experiments.csv")

    # print the names of the sql columns for site_visits
    c.execute("SELECT visit_id, site_rank, site_url FROM site_visits;")

    visits_df = pd.DataFrame(
        c.fetchall(), columns=["visit_id", "site_rank", "site_url"]
    )

    c.execute("SELECT visit_id, url, top_level_url, resource_type FROM http_requests;")
    requests = c.fetchall()
    conn.close()

    # convert to dataframe
    requests_df = pd.DataFrame(
        requests, columns=["visit_id", "url", "top_level_url", "resource_type"]
    )

    # merge with visits
    requests_df = requests_df.merge(visits_df, on="visit_id")
    requests_df.drop("visit_id", axis=1, inplace=True)

    # drop rows where url == site_url or url/ == site_url
    requests_df = requests_df[
        requests_df.apply(
            lambda x: (x.url != x.site_url) and (x.url.rstrip("/") != x.site_url),
            axis=1,
        )
    ]

    requests_df = requests_df.merge(experiments_df, on="site_rank")

    # drop duplicates
    requests_df.drop_duplicates(inplace=True)

    # get top_level_domain
    requests_df["top_level_domain"] = requests_df.url.apply(get_domain)
    requests_df["current_domain"] = requests_df.site_url.apply(get_domain)

    requests_df = requests_df.groupby("id")

    return requests_df


if __name__ == "__main__":
    print("Getting requests...")
    requests = get_requests(__dir__ / "../net_rep-out/easylist/crawl/datadir")
    print("Got requests")

    with DataframeCSVStorageController(
        Path(__dir__ / "../net_rep-out/easylist/results"),
        ["request-blocks.csv", "failed-issues.csv"],
        True,
    ) as storage:
        for issue_id, issue_requests in tqdm(requests, total=len(requests)):
            
            try:
                issue_requests["blocked_before"] = blocked_requests(
                    issue_requests,
                    Path(
                        __dir__
                        / f"../net_rep-out/easylist/filterlists-1/{issue_id}/after_par.txt"
                    ),
                )

                issue_requests["blocked_after"] = blocked_requests(
                    issue_requests,
                    Path(
                        __dir__
                        / f"../net_rep-out/easylist/filterlists-1/{issue_id}/after.txt"
                    ),
                )

                storage.save("request-blocks.csv", issue_requests)

            except Exception as e:
                tb = traceback.format_exc()

                tqdm.write(tb)

                storage.save(
                    "failed-issues.csv",
                    pd.DataFrame(
                        {"issue_id": [issue_id], "error": [str(e)], "traceback": [tb]}
                    ),
                )
