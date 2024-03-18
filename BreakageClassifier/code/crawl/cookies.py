import logging
import sqlite3
from typing import List
from selenium.webdriver import Firefox
from selenium.webdriver.remote.webelement import WebElement
from openwpm.config import BrowserParams, ManagerParams
from openwpm.socket_interface import ClientSocket
from openwpm.commands.types import BaseCommand
from pathlib import Path
import os

base_path = Path(os.path.join(os.path.split(os.path.abspath(__file__))[0]))

with open(str(base_path.joinpath("cookies.js").resolve()), "r") as f:
    COOKIES_SCRIPT = f.read()

DOM_TABLE_NAME = "cookie_banner_evaded"


def create_cookie_banner_table(manager_params: ManagerParams):
    """create the sqlite table for the DOM dump."""
    path = manager_params.data_directory / "crawl-data.sqlite"
    db = sqlite3.connect(path)
    cur = db.cursor()

    cur.execute(
        f"""CREATE TABLE IF NOT EXISTS {DOM_TABLE_NAME} (
            browser_id INTEGER,
            visit_id INTEGER,
            evaded BOOLEAN
            );"""
    )
    cur.close()
    db.close()


def _save_to_db(evaded: bool, browser_id, visit_id, manager_params: ManagerParams):
    # writing the list of nodes to database

    sock = ClientSocket(serialization="dill")
    assert manager_params.storage_controller_address is not None
    sock.connect(*manager_params.storage_controller_address)

    sock.send(
        (
            DOM_TABLE_NAME,
            {
                "browser_id": browser_id,
                "visit_id": visit_id,
                "evaded": evaded,
            },
        )
    )

    sock.close()


def accept_cookies_if_exist(
    webdriver: Firefox, logger: logging.Logger, wait_after_s=2
) -> bool:
    query = webdriver.execute_script(COOKIES_SCRIPT)
    button: WebElement = query["button"]
    iframes: List[WebElement] = query["iframes"]

    # if we found the accept button
    if button is not None:
        try:
            button.click()

            # wait for 2 seconds
            webdriver.execute_async_script(
                f"""
                var done = arguments[0];
                setTimeout(done, {wait_after_s * 1000});
                """
            )

            return True
        except:
            logger.error("Could not click on accept button...")
            return False

    for iframe in iframes:
        if iframe:
            webdriver.switch_to.frame(iframe)
            evaded = accept_cookies_if_exist(webdriver, logger)
            webdriver.switch_to.parent_frame()
            if evaded:
                return True

    return False


class TryEvadeCookiesBannerCommand(BaseCommand):
    """loads the ublock command"""

    def __init__(self) -> None:
        self.logger = logging.getLogger("openwpm")

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "TryEvadeCookiesBannerCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        # webdriver.refresh()

        # sleep(5)
        evaded = accept_cookies_if_exist(webdriver, self.logger)

        if evaded:
            self.logger.info("Evaded cookies banner")
        else:
            self.logger.info("Unable to evade cookies banner or it doesn't exist...")

        _save_to_db(
            evaded, self.browser_id, self.visit_id, manager_params
        )


def clear_cookies(webdriver: Firefox):
    webdriver.get("about:preferences#privacy")
    webdriver.execute_async_script(
        """
        var done = arguments[0];
        setTimeout(done, 2000);
        """
    )
    webdriver.execute_async_script(
        """
        var done = arguments[0];
        SiteDataManager.removeAll().then(done);
        """
    )


class ClearCookiesCommand(BaseCommand):
    """loads the ublock command"""

    def __init__(self) -> None:
        self.logger = logging.getLogger("openwpm")

    # While this is not strictly necessary, we use the repr of a command for logging
    # So not having a proper repr will make your logs a lot less useful
    def __repr__(self) -> str:
        return "ClearCookiesCommand"

    # Have a look at openwpm.commands.types.BaseCommand.execute to see
    # an explanation of each parameter

    def execute(
        self,
        webdriver: Firefox,
        browser_params: BrowserParams,
        manager_params: ManagerParams,
        extension_socket: ClientSocket,
    ) -> None:
        clear_cookies(webdriver)
        self.logger.info("Cleared cookies")
