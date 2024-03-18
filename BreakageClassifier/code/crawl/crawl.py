import argparse
import shutil
import sys
from pathlib import Path
from time import time
from typing import List, Optional
import os
from uuid import uuid4
import numpy as np

import pandas as pd
from openwpm.command_sequence import CommandSequence
from openwpm.commands.browser_commands import GetCommand, SaveScreenshotCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.leveldb import LevelDbProvider
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.task_manager import TaskManager
import http.server
from BreakageClassifier.code.crawl.ablockers import ublock
from .utils import get_ignored_issues, wait_for_val

from Saliency.classify import SaliencyClassifier, SaliencyClassifierConfig

from .ablockers.addon import (
    FILTERLIST_PORT,
    AddonSetupCommand,
    FilterListLoadCommand,
    GetAddonUUIDCommand,
    WaitCommand,
    start_filterlist_server,
)
from .conf import *
from .cookies import (
    ClearCookiesCommand,
    TryEvadeCookiesBannerCommand,
    create_cookie_banner_table,
)
from .dom import DomDumpCommand, SalientDomDumpCommand, create_dom_table
from .error import AddErrorHandlerCommand, HarvestErrorsCommand, create_errors_table
from .interact import (
    SalientRandomInteractCommand,
    SalientRepeatInteractCommand,
    create_interaction_table,
)
from openwpm.errors import CommandExecutionError

from storage_dump.storage import DataframeCSVStorageController

sys.path.insert(1, os.path.join(os.path.split(os.path.abspath(__file__))[0], ".."))
import logging

from ..logger import LOGGER
from tqdm import tqdm

tqdm.pandas()


# constants
VISIT_WATCHDOG = 3 * 60  # seconds


class CrawlerConfig:
    saliency: Optional[SaliencyClassifierConfig] = None
    get_timeout: int = 3 * 120
    dom_dump_timeout: int = 3 * 120
    filterlist_load_timeout: int = 3 * 120
    setup_timeout: int = 5 * 120
    interact_timeout: int = 120
    screenshots = False
    adblocker = ublock
    log_debug = False
    headless = True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Crawler:
    """Crawler class to run experiment crawls."""

    class CrawlerException(Exception):
        pass

    def __init__(
        self,
        num_browsers=1,
        data_dir="./datadir",
        conf: CrawlerConfig = CrawlerConfig(),
        forced=False,
    ):
        try:
            self.logger = logging.getLogger("openwpm")
        except:
            pass

        self.setup_success = None
        self.n_crawls_total = 0
        self.n_crawls_success = 0
        self.n_issues_total = 0
        self.n_issues_success = 0
        self.history = []
        self.data_dir = data_dir
        self.conf = conf
        self._filterlist_server_stop_event = None

        self._curr_issue = None

        # saliency config
        if conf.saliency:
            self.saliency_classifier = SaliencyClassifier(conf.saliency, LOGGER)

        self.NUM_BROWSERS = num_browsers

        # Loads the default ManagerParams
        # and NUM_BROWSERS copies of the default BrowserParams

        self.manager_params = ManagerParams(num_browsers=self.NUM_BROWSERS)
        self.browser_params = [
            BrowserParams(display_mode="headless" if conf.headless else "native")
            for _ in range(self.NUM_BROWSERS)
        ]

        # Update browser configuration (use this for per-browser settings)
        for browser_param in self.browser_params:
            # Record HTTP Requests and Responses
            browser_param.http_instrument = True
            # Record cookie changes
            browser_param.cookie_instrument = True
            # Record Navigations
            browser_param.navigation_instrument = True

            if conf.headless:
                # Record JS Web API calls
                browser_param.js_instrument = True
                # Record the callstack of all WebRequests made
                browser_param.callstack_instrument = True
            else:
                # if not headless -> FinalizeCommand Hangs Forever
                # common issue with OpenWPM: https://github.com/openwpm/OpenWPM/issues/947

                # Record JS Web API calls
                browser_param.js_instrument = False
                # Record the callstack of all WebRequests made
                browser_param.callstack_instrument = False

            # Record DNS resolution
            browser_param.dns_instrument = True
            # save the javascript files
            browser_param.save_content = "script"

            browser_param.console_log = True

            if conf.headless:
                browser_param.js_instrument_settings = [
                    "collection_fingerprinting",
                    {
                        "window": {
                            "propertiesToInstrument": [
                                "name",
                                "localStorage",
                                "sessionStorage",
                                "onerror",
                            ],
                            "nonExistingPropertiesToInstrument": [
                                "last_log_event",
                            ],
                        }
                    },
                    {"window.console": {"logCallStack": True}},
                    {
                        "window.document": {
                            "propertiesToInstrument": {
                                "getElementById",
                                "getElementsByClassName",
                                "getElementsByTagName",
                                "querySelector",
                                "querySelectorAll",
                            },
                            "logCallStack": True,
                        }
                    },
                ]

        # Update TaskManager configuration (use this for crawl-wide settings)
        self.manager_params.data_directory = Path(data_dir)
        self.manager_params.log_path = Path(f"{data_dir}/openwpm.log")

        # if forced delete the folder
        if forced:
            shutil.rmtree(
                str(self.manager_params.data_directory.resolve().absolute()),
                ignore_errors=True,
            )

        self.site_rank_offset = 0
        if os.path.exists(f"{data_dir}/experiments.csv") and not forced:
            experiments = pd.read_csv(f"{data_dir}/experiments.csv")
            self.site_rank_offset = (
                experiments.site_rank.max() + 1 if len(experiments) > 0 else 0
            )

    def setup_crawl_env(self, manager: TaskManager, filterlists_dir: Path):
        def callback(success: bool):
            self.setup_success = success

        # add extra tables to the db
        create_dom_table(self.manager_params)
        create_interaction_table(self.manager_params)
        create_errors_table(self.manager_params)
        create_cookie_banner_table(self.manager_params)

        # turn on filterlists http server
        # self._filterlist_server_stop_event = start_filterlist_server(filterlists_dir)
        # self.logger.info("Started filterlist server at %s", FILTERLIST_PORT)

        # load the ublock extension
        command_sequence = CommandSequence(
            "about:blank", site_rank=-1, callback=callback, blocking=True
        )

        for i in range(self.NUM_BROWSERS):
            command_sequence.append_command(
                self.conf.adblocker.AddonSetupCommand(),
                timeout=self.conf.setup_timeout,
            )

            command_sequence.append_command(
                GetCommand(url="about:blank", sleep=3), timeout=self.conf.get_timeout
            )

            manager.execute_command_sequence(command_sequence, index=i)

            if not wait_for_val(lambda: self.setup_success):
                raise self.CrawlerException("Could not setup crawl environment")

    def crawl_visit(
        self,
        issue: pd.Series,
        filterlist_path: Path,
        filterlist_tag: str,
        is_first_visit: bool,
        manager: TaskManager,
        exp_log_storage: DataframeCSVStorageController,
    ):
        """Performs one crawl task given a site and a filter list.

        Args:
            metadata (pd.Series): Information about the site
            filterlist_path (Path): Path to filterlist
            manager (TaskManager): openwpm task manager
        """
        self._curr_issue = issue

        # get the site url

        site = issue.url
        site_rank = len(self.history) + self.site_rank_offset

        # crawling section of url

        # template for the returned output
        res = {
            "id": issue.id,
            "site_rank": site_rank,
            "filterlist": filterlist_tag,
            "success": False,
            "completed": False,
        }

        def callback(success: bool, val: str = site) -> None:
            res["success"] = success
            res["completed"] = True
            self.history.append(res)

            if success:
                exp_log_storage.save("experiments.csv", pd.DataFrame([res]))

        # Parallelize sites over all number of browsers set above.
        command_sequence = CommandSequence(
            site, site_rank=site_rank, callback=callback, blocking=True, retry_number=1
        )

        command_sequence.append_command(
            GetAddonUUIDCommand(
                self.conf.adblocker.ID,
            ),
            timeout=30,
        )

        command_sequence.append_command(ClearCookiesCommand(), timeout=30)

        command_sequence.append_command(
            self.conf.adblocker.FilterListLoadCommand(path=filterlist_path),
            timeout=self.conf.filterlist_load_timeout,
        )

        # Start by visiting the page
        command_sequence.append_command(
            GetCommand(url=site, sleep=3), timeout=self.conf.get_timeout
        )
        command_sequence.append_command(WaitCommand(30), timeout=120)
        # command_sequence.append_command(WaitCommand(15), timeout=120)

        # try to mitigate cookies
        command_sequence.append_command(TryEvadeCookiesBannerCommand(), timeout=30)

        # a page refresh happens here

        # [TO REMOVE] Error JS approach replaced by instrumentation
        # add error instrumentation
        # command_sequence.append_command(AddErrorHandlerCommand())

        if self.conf.screenshots:
            command_sequence.append_command(
                SaveScreenshotCommand(f"{filterlist_tag}-base")
            )

        if self.conf.saliency is not None:
            if is_first_visit:
                command_sequence.append_command(
                    SalientDomDumpCommand(
                        self.saliency_classifier,
                        screenshot_suffix=f"{filterlist_tag}-salient"
                        if self.conf.screenshots
                        else None,
                    ),
                    timeout=self.conf.dom_dump_timeout,
                )

                command_sequence.append_command(
                    SalientRandomInteractCommand(), timeout=self.conf.interact_timeout
                )
            else:
                command_sequence.append_command(DomDumpCommand())
                command_sequence.append_command(
                    SalientRepeatInteractCommand(), timeout=self.conf.interact_timeout
                )
        else:
            command_sequence.append_command(DomDumpCommand())

        # [TO REMOVE] Error JS approach replaced by instrumentation
        # command_sequence.append_command(HarvestErrorsCommand())
        num_processed = len(self.history)

        try:
            manager.execute_command_sequence(command_sequence)
        except CommandExecutionError as e:
            self.logger.error(str(e))
            raise self.CrawlerException("Visit Command Sequence Failed")

        if (
            wait_for_val(lambda: res["completed"], False)
            and len(self.history) <= num_processed
        ):
            # manager.close(relaxed=False)
            self.logger.warn("Crawl timed out")
            res["success"] = False
            # add to the history
            self.history.append(res)

            is_success = False
        else:
            is_success = self.history[site_rank - self.site_rank_offset]["success"]

        return is_success

    def crawl_site(
        self,
        manager: TaskManager,
        issue: pd.Series,
        filterlists_dir: Path,
        exp_log_storage: DataframeCSVStorageController,
        previous_session_sites: pd.DataFrame,
        filterlist_tags: List[str] = ["after", "before", None],
    ):
        """performs crawling tasks on a site of interest. once without ad blocker, once with the breaking list and once with the unbroken

        Args:
            metadata (pd.Series):  Information about the site
            filterlists_path (Path): Path to the folder containing the filterlists
            manager (TaskManager): openwpm task manager
        """

        if issue.id in previous_session_sites:
            self.logger.info(
                f"issue {issue.id} skipped: Already processed in previous session"
            )
            return issue.id, True

        is_success = False

        self.n_issues_total += 1

        try:
            for i, tag in enumerate(filterlist_tags):
                self.n_crawls_total += 1

                filterlist_path = (
                    filterlists_dir / f"{issue.id}/{tag}.txt" if tag else None
                )

                is_success = self.crawl_visit(
                    issue,
                    filterlist_path,
                    tag,
                    i == 0,
                    manager,
                    exp_log_storage,
                )

                self.logger.info(
                    "issue %s [%i/%i]: %s filterlist crawl %s | %s",
                    issue.id,
                    i + 1,
                    len(filterlist_tags),
                    tag,
                    "success" if is_success else "failed. Skipping issue...",
                    issue.url,
                )

                if not is_success:
                    break

                self.n_crawls_success += 1

            # if we reach this point before an exception was raised, the crawl was successfull
            if is_success:
                self.n_issues_success += 1

        except Crawler.CrawlerException as e:
            # if the crawl failed
            self.logger.error(str(e))
            self.logger.info("Crawl failed. Skipping issue...")
            is_success = False

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
        start_after_issue=None,
        ignore: Path = None,
    ):
        """performs crawling tasks on the different sample sites from forum dataset.

        Args:
            metadata (Path): path to the metadata csv file
            filterlists_path (Path): path to the folder of filterlists
        """

        # issues to ignore
        ignored = get_ignored_issues(ignore)

        # loading the data
        df = pd.read_csv(issues_path)

        # check if url or test_url

        if "test_url" in df.columns:
            df["url"] = df["test_url"]

        # getting labels

        df = df[
            (df["should_include"] == True)
            & (~df["before_commit"].isnull())
            & (~df["id"].isin(ignored))
        ]

        if start_after_issue is not None:
            start_index = df[df["id"] == start_after_issue].index[0]
            df = df.iloc[start_index + 1 :]

        # getting the number of sites to crawl
        start = 0 if start is None else start
        num = len(df) - start if num is None else num
        num = len(df) - start if start + num > len(df) else num

        # setting the outputs
        site_success = pd.DataFrame(columns=["id", "success"])

        # read from previous run
        try:
            site_success = pd.read_csv(self.data_dir + "/experiments.csv")
            self.history.extend(site_success.to_dict("list"))

            # a site is successfull if we crawled the last visit with no filtelist
            site_success = site_success[site_success["filterlist"].isna()][
                ["id", "success"]
            ].drop_duplicates()
            self.logger.info("Loaded previous crawl session...")

        except FileNotFoundError:
            self.logger.info("Could not read previous crawls. Starting new crawl...")

        # start the crawler
        with DataframeCSVStorageController(
            Path(f"{self.data_dir}"), ["experiments.csv"], True
        ) as exp_log_storage:
            processed_sites = []

            for _, issue in df.iloc[np.arange(start, start + num)].iterrows():
                with TaskManager(
                    self.manager_params,
                    self.browser_params,
                    SQLiteStorageProvider(Path(f"{self.data_dir}/crawl-data.sqlite")),
                    LevelDbProvider(Path(f"{self.data_dir}/content.ldb")),
                    logger_kwargs={
                        "log_level_console": logging.DEBUG
                        if self.conf.log_debug
                        else logging.INFO
                    },
                ) as manager:
                    self.setup_crawl_env(manager, filterlists_path)
                    _, success = self.crawl_site(
                        manager,
                        issue,
                        filterlists_path,
                        exp_log_storage,
                        set(site_success["id"]),
                    )

                    processed_sites.append({"id": issue.id, "success": success})

            processed_sites = pd.DataFrame(processed_sites)

            # stop the filterlist server
            if self._filterlist_server_stop_event:
                self._filterlist_server_stop_event.set()

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

        with open(self.data_dir + "/failed.txt", "w") as f:
            f.writelines([str(x) + "\n" for x in failed_sites])

    def debug_issue(
        self,
        issues_path: Path,
        filterlists_path: Path,
        issue: int,
    ):
        """performs crawling tasks on the different sample sites from forum dataset.

        Args:
            metadata (Path): path to the metadata csv file
            filterlists_path (Path): path to the folder of filterlists
        """

        # loading the data
        df = pd.read_csv(issues_path)

        # check if url or test_url

        if "test_url" in df.columns:
            df["url"] = df["test_url"]

        # getting labels

        df = df[(df["should_include"] == True) & (~df["before_commit"].isnull())]

        try:
            issue_info = df[df["id"] == issue].iloc[0]
        except IndexError:
            self.logger.error(f"Could not find issue {issue}")
            return

        # start the crawler
        with DataframeCSVStorageController(
            Path(f"{self.data_dir}"), ["experiments.csv"], True
        ) as exp_log_storage, TaskManager(
            self.manager_params,
            self.browser_params,
            SQLiteStorageProvider(Path(f"{self.data_dir}/crawl-data.sqlite")),
            LevelDbProvider(Path(f"{self.data_dir}/content.ldb")),
            logger_kwargs={
                "log_level_console": logging.DEBUG
                if self.conf.log_debug
                else logging.INFO
            },
        ) as manager:
            self.setup_crawl_env(manager, filterlists_path)

            self.crawl_site(manager, issue_info, filterlists_path, exp_log_storage, {})

            # stop the filterlist server
            if self._filterlist_server_stop_event:
                self._filterlist_server_stop_event.set()
