import sqlite3
from time import time
import pandas as pd
from pathlib import Path
from openwpm.command_sequence import CommandSequence
from openwpm.commands.browser_commands import GetCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.leveldb import LevelDbProvider
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager
from tqdm import tqdm
from storage_dump.storage import DataframeCSVStorageController

from BreakageClassifier.code.crawl.ablockers.addon import (
    FilterListLoadCommand,
    AddonSetupCommand,
    WaitCommand,
)
from BreakageClassifier.code.crawl.cookies import ClearCookiesCommand
from BreakageClassifier.code.crawl.conf import UBLOCK_XPI_PATH
import logging
import BreakageClassifier.code.logger as logger

VISIT_WATCHDOG = 3 * 60  # seconds

tqdm.pandas()


class Crawler:
    class CrawlerException(Exception):
        pass

    def __init__(
        self,
        num_browsers=1,
        data_dir: Path = Path("./datadir"),
    ):
        try:
            self.logger = logging.getLogger("openwpm")
        except:
            pass

        self.n_crawls_total = 0
        self.n_crawls_success = 0
        self.n_issues_total = 0
        self.n_issues_success = 0
        self.history = []
        self.data_dir = data_dir

        self.NUM_BROWSERS = num_browsers

        # Loads the default ManagerParams
        # and NUM_BROWSERS copies of the default BrowserParams

        self.manager_params = ManagerParams(num_browsers=self.NUM_BROWSERS)
        self.browser_params = [
            BrowserParams(display_mode="headless") for _ in range(self.NUM_BROWSERS)
        ]

        # Update browser configuration (use this for per-browser settings)
        for browser_param in self.browser_params:
            # Record HTTP Requests and Responses
            browser_param.http_instrument = True
            # Record cookie changes
            browser_param.cookie_instrument = True
            # Record Navigations
            browser_param.navigation_instrument = True
            # Record JS Web API calls
            browser_param.js_instrument = True
            # Record the callstack of all WebRequests made
            browser_param.callstack_instrument = True
            # Record DNS resolution
            browser_param.dns_instrument = True
            # save the javascript files
            # browser_param.save_content = "script"

            browser_param.console_log = True

            browser_param.js_instrument_settings = [
                "collection_fingerprinting",
            ]

        # Update TaskManager configuration (use this for crawl-wide settings)
        self.manager_params.data_directory = Path(data_dir)
        self.manager_params.log_path = Path(f"{data_dir}/openwpm.log")

    def crawl_visit(
        self,
        metadata: pd.Series,
        filterlist_path: Path,
        filterlist_tag: str,
        manager: TaskManager,
        exp_log_storage: DataframeCSVStorageController,
    ):
        """Performs one crawl task given a site and a filter list.

        Args:
            metadata (pd.Series): Information about the site
            filterlist_path (Path): Path to filterlist
            manager (TaskManager): openwpm task manager
        """

        # get the site url

        site = metadata.url
        site_rank = len(self.history)

        # crawling section of url

        # template for the returned output
        res = {
            "id": metadata.id,
            "site_rank": site_rank,
            "filterlist": filterlist_tag,
            "success": False,
        }

        def callback(success: bool, val: str = site) -> None:
            res["success"] = success
            self.history.append(res)

            if success:
                exp_log_storage.save("experiments.csv", pd.DataFrame([res]))

        # Parallelize sites over all number of browsers set above.
        command_sequence = CommandSequence(
            site, site_rank=site_rank, callback=callback, reset=False
        )

        # command_sequence.append_command(ClearCookiesCommand(), timeout=30)

        # command_sequence.append_command(
        #     FilterListLoadCommand(path=filterlist_path, check=False)
        # )

        # Start by visiting the page
        command_sequence.append_command(GetCommand(url=site, sleep=3), timeout=120)
        command_sequence.append_command(WaitCommand(20), timeout=120)

        # wait for 30 seconds

        manager.execute_command_sequence(command_sequence)

        t_start = time()

        # while the job is not finished
        while site_rank >= len(self.history) and time() - t_start < VISIT_WATCHDOG:
            pass

        # if the process timed out
        if site_rank >= len(self.history):
            manager.close(relaxed=False)
            self.logger.info("Crawl timed out")
            res["success"] = False
            # add to the history
            self.history.append(res)

            is_success = False
        else:
            is_success = self.history[site_rank]["success"]

        if not is_success:
            raise Crawler.CrawlerException(
                f"Failed to crawl issue={metadata.id}, url={site}, filterlist={filterlist_tag}"
            )

        return True

    def crawl_site(
        self,
        manager: TaskManager,
        issue: pd.Series,
        filterlists_dir: Path,
        exp_log_storage: DataframeCSVStorageController,
        previous_session_sites: pd.DataFrame,
        filterlist_tags: list = [
            None,
        ],
    ):
        # check if the site was already processed in the previous session
        if issue.id in previous_session_sites:
            self.logger.info(
                f"issue {issue.id} skipped: Already processed in previous session"
            )
            return issue.id, True

        is_success = False

        # if the site was not processed in the previous session

        self.n_issues_total += 1

        # crawl the site with the filterlists
        try:
            for tag in filterlist_tags:
                self.n_crawls_total += 1

                if tag is not None:
                    filterlist_path = filterlists_dir + f"/{issue.id}/{tag}.txt"
                else:
                    filterlist_path = None

                is_success = self.crawl_visit(
                    issue,
                    filterlist_path,
                    tag,
                    manager,
                    exp_log_storage,
                )

                self.logger.info(
                    "issue %s [1/%i]: {%s} filterlist crawl %s | %s",
                    issue.id,
                    len(filterlist_tags),
                    tag,
                    "success" if is_success else "failed. Skipping issue...",
                    issue.url,
                )

                # if the crawl was successfull
                self.n_crawls_success += 1

            self.n_issues_success += 1

        except Crawler.CrawlerException as e:
            # if the crawl failed
            self.logger.error(str(e))
            self.logger.info("Crawl failed. Skipping issue...")
            is_success = False

        # print the stats
        self.logger.info(
            "Stats: successfull crawls - %i/%i ~ %i%% | successfull issues - %i/%i ~ %i%%"
            % (
                self.n_crawls_success,
                self.n_crawls_total,
                self.n_crawls_success * 100 // self.n_crawls_total,
                self.n_issues_success,
                self.n_issues_total,
                self.n_issues_success * 100 // self.n_issues_total,
            )
        )

        return issue.id, is_success

    def crawl_from_dataset(
        self,
        issues_path: Path,
        filterlists_path: Path,
        num=None,
        start=None,
        ignore: Path = None,
        debug_issue=None,
    ):
        """performs crawling tasks on the different sample sites from forum dataset.

        Args:
            metadata (Path): path to the metadata csv file
            filterlists_path (Path): path to the folder of filterlists
        """
        issues_path = str(Path(issues_path).absolute())

        filterlists_path = str(Path(filterlists_path).absolute())

        # issues to ignore
        ignored = []

        if ignore:
            with open(str(ignore.resolve()), "r") as f:
                ignored = f.readlines()
            ignored = [int(x) for x in ignored]

        # loading the data
        df = pd.read_csv(issues_path)

        # check if url or test_url

        if "test_url" in df.columns:
            df["url"] = df["test_url"]

        # getting labels

        # df = df[
        #     ((df["label"] == "D") | (df["label"] == "S"))
        #     & (~df["before_commit"].isnull())
        #     & (~df["id"].isin(ignored))
        # ]

        # if debug_issue:
        #     df = df[df["id"] == debug_issue]

        # setting the bounds
        if start is None:
            start = 0

        if num is None:
            num = len(df) - start

        if start + num > len(df):
            num = len(df) - start

        # setting the outputs
        site_success = pd.DataFrame(columns=["id", "success"])

        # read from previous run
        try:
            site_success = pd.read_csv(self.data_dir / "experiments.csv")
            self.history.extend(site_success.to_dict("list"))

            # a site is successfull if we crawled the last visit with no filtelist
            site_success = site_success[site_success["filterlist"].isna()][
                ["id", "success"]
            ].drop_duplicates()
            print("Loaded previous crawl session...")
        except Exception as e:
            print("Could not read previous crawls. Starting new crawl...")

        with DataframeCSVStorageController(
            Path(f"{self.data_dir}"), ["experiments.csv"], True
        ) as exp_log_storage, TaskManager(
            self.manager_params,
            self.browser_params,
            SQLiteStorageProvider(self.data_dir / "crawl-data.sqlite"),
            LevelDbProvider(self.data_dir / "content.ldb"),
        ) as manager:
            # install addon
            # command_sequence = CommandSequence(
            #     "about:blank",
            #     site_rank=-1,
            #     callback=lambda success: None,
            #     reset=False,
            # )

            # command_sequence.append_command(
            #     AddonSetupCommand(
            #         path=str(Path(UBLOCK_XPI_PATH).absolute()), check=False
            #     )
            # )

            # manager.execute_command_sequence(command_sequence)

            processed_sites = pd.DataFrame(columns=["session", "id", "success"])

            processed_sites[["id", "success"]] = df.iloc[
                list(range(start, start + num))
            ].progress_apply(
                lambda row: self.crawl_site(
                    manager,
                    row,
                    filterlists_path,
                    exp_log_storage,
                    set(site_success["id"]),
                ),
                axis=1,
                result_type="expand",
            )

            failed_sites_now = set(
                processed_sites[~processed_sites["success"]].id.values.tolist()
            )

            if len(site_success) != 0:
                failed_sites_before = set(
                    site_success[~site_success["success"]].id.values.tolist()
                )
                failed_sites = failed_sites_now | failed_sites_before
            else:
                failed_sites = failed_sites_now

        with open(self.data_dir / "failed.txt", "w") as f:
            f.writelines([str(x) + "\n" for x in failed_sites])


if __name__ == "__main__":
    crawler = Crawler(
        data_dir=Path("../net_rep-out/easylist/crawl/datadir"),
    )

    # Crawl with no filterlists
    crawler.crawl_from_dataset(
        issues_path=Path("../net_rep-out/easylist/easylist-data-rep.csv"),
        filterlists_path=Path("../net_rep-out/easylist/filterlists"),
    )
